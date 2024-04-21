CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nproc_per_node=8
IMAGENET_PATH=/path/to/ImageNet

base_dir=/path/to/AdvXL_Release
output_dir=${base_dir}/output/sota/h14/336
#mkdir -p ${output_dir}
checkpoint=advxl_vit_h14.pth

norm=Linf
img_size=336
batch_size=5
model_name=vit_huge_patch14_224
extra_params="--gp avg --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 --norm ${norm}"

state_path=img${img_size}_${norm}_autoattack_state.json
#rm ${output_dir}/${state_path}.*

cd ${base_dir}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --nproc_per_node=${nproc_per_node} \
--rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
eval_autoattack.py ${IMAGENET_PATH} --split validation \
--model ${model_name} \
--state-path ${output_dir}/${state_path} \
--checkpoint ${output_dir}/${checkpoint} \
--batch-size ${batch_size} --pin-mem --log-freq 50 \
--workers 8 --img-size ${img_size} --model-img-size ${img_size} \
$extra_params 2>&1 | tee ${output_dir}/output_img${img_size}_${norm}.txt