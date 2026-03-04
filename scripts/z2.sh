#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate bishe
# cd Bishe_2

# 从外部传入可见的GPU，默认为0
CUDA_VISIBLE_DEVICES=${1:-0}
export CUDA_VISIBLE_DEVICES

dataset_name="AppUT-appnon2new"
LOG_FILE="logs/finetune/z2-${dataset_name}-$(date +%s)_$$.txt"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[EXIT] $(date -Is) inference finished, exit_code=$rc, log=$LOG_FILE"' EXIT

LOG_FILE="logs/finetune/${dataset_name}-$(date +%s).txt"
# sudo ln -s /usr/lib/x86_64-linux-gnu/libc.a /usr/lib/x86_64-linux-gnu/liblibc.a;
# python --version;
EVAL_EPOCHS="2 4 6 8"
python -m z1.framework \
    --finetune_mode \
    --split_layers_num=20 \
    --z2_mode \
    --gh=gh++ \
    --dom_loss=lmmd \
    --single_gpu \
    --amp \
    --amp_dtype=bf16 \
    --flash_attn \
    --memory_bank \
    --queue_size=128 \
    --eval_epochs $EVAL_EPOCHS \
    --resume_encoder=models/encoder/90000/best_checkpoint.pt \
    --resume_lora0=models/alignment1/3mixed/300/best_checkpoint.pt \
    --resume_linear=models/alignment2/3mixed/1800-500/best_checkpoint.pt \
    --projector=linear \
    --linear_output_dim=4096 \
    --train_dir="datasets/finetuning/5/$dataset_name/train" \
    --train_dir_t="datasets/finetuning/5/$dataset_name/tgt" \
    --test_dir="datasets/finetuning/5/$dataset_name/test" \
    --save_dir="models/finetuning/${dataset_name}-5" \
    --llm="Qwen3-VL-8B-Instruct" \
    --nodistributed \
    --log_freq=16 \
    --batch_size=1 \
    --batch_size_t=1 \
    --accumulation_steps=16 \
    --epochs=10 