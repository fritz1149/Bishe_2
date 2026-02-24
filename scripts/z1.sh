source /opt/anaconda3/etc/profile.d/conda.sh
conda activate bishe
# cd Bishe_2

LOG_FILE="logs/finetune/$(date +%s)_$$.txt"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[EXIT] $(date -Is) inference finished, exit_code=$rc, log=$LOG_FILE"' EXIT

dataset_name="App53-Time"
LOG_FILE="logs/finetune/${dataset_name}-$(date +%s).txt"
# sudo ln -s /usr/lib/x86_64-linux-gnu/libc.a /usr/lib/x86_64-linux-gnu/liblibc.a;
# python --version;
python -m z1.framework \
    --finetune_mode \
    --split_layers_num=20 \
    --amp \
    --amp_dtype=bf16 \
    --stop_epochs=4 \
    --eval_epochs=4 \
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
    --log_freq=20 \
    --batch_size=2 \
    --accumulation_steps=8 \
    --epochs=10 