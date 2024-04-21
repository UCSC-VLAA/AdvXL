 # Revisiting Adversarial Training at Scale

This is the official repository for our paper: "Revisiting Adversarial Training at Scale."


## Abstract:
The machine learning community has witnessed a drastic change in the training pipeline, pivoted by those ``foundation models'' with unprecedented scales. However, the field of adversarial training is lagging behind, predominantly centered around small model sizes like ResNet-50, and tiny and low-resolution datasets like CIFAR-10. To bridge this transformation gap, this paper provides a modern re-examination with adversarial training, investigating its potential benefits when applied at scale. Additionally, we introduce an efficient and effective training strategy to enable adversarial training with giant models and web-scale data at an affordable computing cost. We denote this newly introduced framework as AdvXL.

Empirical results demonstrate that AdvXL establishes new state-of-the-art robust accuracy records under AutoAttack on ImageNet-1K. For example, by training on DataComp-1B dataset, our AdvXL empowers a vanilla ViT-g model to substantially surpass the previous records of $l_{\infty}$-, $l_{2}$-, and $l_{1}$-robust accuracy by margins of **11.4%**, **14.2%** and **12.9%**, respectively. This achievement posits AdvXL as a pioneering approach, charting a new trajectory for the efficient training of robust visual representations at significantly larger scales. 


<div align="center">
  <img src="figures/advxl_scale.png"/>
</div>
<div align="center">
  <img src="figures/advxl_performance.png"/>
</div>

## Installation
Installation and preparation follow the [TIMM Repo](https://github.com/huggingface/pytorch-image-models).
Additionally, [RobustBench](https://github.com/RobustBench/robustbench) is needed to evaluate model robustness.
We also provide a sample conda environment yml file [here](environment.yml), that we used to reproduce the eval results.


## Usage
### Testing Instructions
For robustness under PGD attack, use `validate.py`.
For robustness under AutoAttack, use `eval_autoattack.py`.

We have also provided some example eval scripts under `scripts/eval`. 
Put the proper weights under `${output_dir}/${checkpoint}`, and they should be able to readily reproduce the results reported in our paper.

# TODO: upload model to gdrive or huggingface
### Model Weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Dataset</th>
<th valign="bottom">Sample@Resolution</th>
<th valign="bottom">Adv. Steps</th>
<th valign="bottom">Clean</th>
<th valign="bottom">Linf</th>
<th valign="bottom">L2</th>
<th valign="bottom">L1</th>
<th valign="bottom">Weights</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-H/14</td>
<td align="center">DataComp-1B + ImageNet-1K</td>
<td align="center">5.12B@84 + 38.4M@224 + 6.4M@336</td>
<td align="center">2/3</td>
<td align="center">83.9</td>
<td align="center">69.8</td>
<td align="center">69.8</td>
<td align="center">46.0</td>
<td align="center"><a href="https://huggingface.co/UCSC-VLAA/AdvXL-ViT-H14/blob/main/advxl_vit_h14.pth">download</td>
<tr><td align="left">ViT-g/14</td>
<td align="center">DataComp-1B + ImageNet-1K</td>
<td align="center">5.12B@84 + 38.4M@224 + 6.4M@336</td>
<td align="center">2/3</td>
<td align="center">83.9</td>
<td align="center">71.0</td>
<td align="center">70.4</td>
<td align="center">46.7</td>
<td align="center"><a href="https://huggingface.co/UCSC-VLAA/AdvXL-ViT-g14/blob/main/advxl_vit_g14.pth">download</td>
</tbody></table>


## License
This project is under the Apache 2.0 License.


## Acknowledgement
This repo is heavily based on [TIMM](https://github.com/huggingface/pytorch-image-models).
Many thanks to the awesome works from the open-source community!

This work is partially supported by a gift from Open Philanthropy. 
We also thank Center for AI Safety, TPU Research  Cloud (TRC) program, and Google Cloud Research Credits program for supporting our computing needs.

