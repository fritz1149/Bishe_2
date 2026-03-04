#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate bishe
# cd Bishe_2
dataset_names=(
    "App53-Time"
)
CUDA_VISIBLE_DEVICES=${1:-0}
export CUDA_VISIBLE_DEVICES



for dataset_name in "${dataset_names[@]}"; do
    LOG_FILE="logs/finetune/${dataset_name}-$(date +%s)_$$.txt"
    mkdir -p "$(dirname "$LOG_FILE")"
    (
        trap 'rc=$?; echo "[EXIT] $(date -Is) inference finished, exit_code=$rc, log=$LOG_FILE"' EXIT
        # sudo ln -s /usr/lib/x86_64-linux-gnu/libc.a /usr/lib/x86_64-linux-gnu/liblibc.a;
        # python --version;
        python -m z1.framework \
            --finetune_mode \
            --split_layers_num=20 \
            --single_gpu \
            --classifier_mode \
            --amp \
            --amp_dtype=bf16 \
            --flash_attn \
            --base_lr=1e-4 \
            --resume_encoder=models/encoder/90000/best_checkpoint.pt \
            --resume_lora0=models/alignment1/3mixed/300/best_checkpoint.pt \
            --resume_linear=models/alignment2/3mixed/1800-500/best_checkpoint.pt \
            --projector=linear \
            --linear_output_dim=4096 \
            --train_dir="datasets/finetuning/5/$dataset_name/train" \
            --test_dir="datasets/finetuning/5/$dataset_name/test" \
            --save_dir="models/finetuning/${dataset_name}-5" \
            --llm="Qwen3-VL-8B-Instruct" \
            --nodistributed \
            --log_freq=32 \
            --batch_size=1 \
            --accumulation_steps=16 \
            --epochs=10
    ) > >(tee -a "$LOG_FILE") 2>&1
done