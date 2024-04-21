#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)

Modified by Zeyu to evaluate robustness under autoattack
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from collections import OrderedDict
from pathlib import Path

from timm.bits import initialize_device, Tracker, Monitor, AccuracyTopK, AvgTensor
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_transform_v2, create_loader_v2, resolve_data_config, RealLabelsImagenet, \
    PreprocessCfg
from timm.utils import natural_key, setup_default_logging, random_seed
from timm.adversarial import NoOpAttacker, PGDAttacker, AdvModel
from timm.adversarial.autoattack import AutoAttack
import robustbench

_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--model-img-size', type=int, default=None, metavar='N',
                    help='Image patch size for model (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=20, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# parser.add_argument('--num-gpu', type=int, default=1,
#                     help='Number of GPUS to use')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--state-path', default=None, type=str, metavar='FILENAME',
                    help='path to store autoattack state')
parser.add_argument('--force-cpu', action='store_true', default=False,
                    help='Force CPU to be used even if HW accelerator exists.')


########################
parser.add_argument('--full-val-set', action='store_true', default=False,
                    help='use full imagenet validation set')
parser.add_argument("--norm", choices=["Linf", "L2", "L1"],
                    default="Linf", help="choose attack norm. available choices: l_inf, l2, l1 ")
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')

parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
########################

class normalize_model():
    def __init__(self, model, norm_func):
        self.model = model
        self.norm_func = norm_func

    def __call__(self, x):
        return self.model(self.norm_func(x))


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint

    dev_env = initialize_device(force_cpu=args.force_cpu, amp=args.amp)

    # create model
    model_kwargs = dict()
    if args.model_img_size is not None:
        model_kwargs['img_size'] = int(args.model_img_size)
    print(f'**kwargs in create_model: {model_kwargs}')

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        **model_kwargs)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    criterion = nn.CrossEntropyLoss()

    model, criterion = dev_env.to_device(model, criterion)
    model.to(dev_env.device)
    model.eval()

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    eval_pp_cfg = PreprocessCfg(
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        crop_pct=data_config['crop_pct'],
        # use non-op norm in transform, and normalize input right before model
        mean=(0., 0., 0.),
        std=(1., 1., 1.),
    )

    if args.full_val_set:
        dataset = create_dataset(
            root=args.data, name=args.dataset, split=args.split,
            download=args.dataset_download, load_bytes=args.tf_preprocessing, class_map=args.class_map)
        test_loader = create_loader_v2(
            dataset,
            batch_size=args.batch_size,
            is_training=False,
            pp_cfg=eval_pp_cfg,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            normalize_in_transform=True,
            persistent_workers=False)

        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)
    else:
        transform = create_transform_v2(
            cfg=eval_pp_cfg,
            is_training=False,
            normalize=True,
            separate=False,
        )
        x_test, y_test = robustbench.data.load_imagenet(5000, data_dir=args.data, transforms_test=transform)
        # select samples for each rank
        x_test = x_test[dev_env.global_rank::dev_env.world_size]
        y_test = y_test[dev_env.global_rank::dev_env.world_size]

    ### Evaluate AutoAttack ###
    _logger.info(f'start evaling {len(y_test)} number of samples in rank: {dev_env.global_rank}')
    start_timestamp = time.perf_counter()

    # make autoattack code happy
    mean_tensor = dev_env.to_device(torch.Tensor(data_config['mean'])).view(1, 3, 1, 1)
    std_tensor = dev_env.to_device(torch.Tensor(data_config['std'])).view(1, 3, 1, 1)
    norm_func = lambda x: (x - mean_tensor) / std_tensor
    wrapped_norm_model = normalize_model(model, norm_func)

    if args.norm == 'Linf':
        epsilon = 4 / 255.
    elif args.norm == 'L2':
        epsilon = 2.
    elif args.norm == 'L1':
        epsilon = 75.
    else:
        raise NotImplementedError
    adversary = AutoAttack(wrapped_norm_model, norm=args.norm, eps=epsilon, version='standard')
    robust_accuracy = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size, state_path=Path(args.state_path + f'.{dev_env.global_rank}'))
    # TODO: test barrier here
    dev_env.barrier()
    robust_accuracy = torch.tensor(robust_accuracy, device=dev_env.device)
    robust_accuracy_average = dev_env.all_reduce_(robust_accuracy, average=True).item()
    _logger.info(f'device: {dev_env.global_rank}, average autoattack robust accuracy: {robust_accuracy_average}')
    end_timestamp = time.perf_counter()
    _logger.info(f'eval took {end_timestamp - start_timestamp} seconds')



if __name__ == '__main__':
    setup_default_logging()
    args = parser.parse_args()
    validate(args)
