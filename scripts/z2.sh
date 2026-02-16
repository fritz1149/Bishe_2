source /work/miniconda3/etc/profile.d/conda.sh
conda activate bishe
# cd Bishe_2

LOG_FILE="logs/finetune/$(date +%s)_$$.txt"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[EXIT] $(date -Is) inference finished, exit_code=$rc, log=$LOG_FILE"' EXIT

dataset_name_s="App53-Time"
dataset_name_t="App53->Time"
LOG_FILE="logs/finetune/${dataset_name_s}-${dataset_name_t}-$(date +%s).txt"
# sudo ln -s /usr/lib/x86_64-linux-gnu/libc.a /usr/lib/x86_64-linux-gnu/liblibc.a;
# python --version;
python -m z1.framework \
    --finetune_mode \
    --split_layers_num=20 \
    --z2_mode \
    --gh=deactivated \
    --dom_loss=lmmd \
    --amp \
    --amp_dtype=bf16 \
    --memory_bank \
    --queue_size=64 \
    --momentum_k=0.999 \
    --resume_encoder=models/encoder/90000/best_checkpoint.pt \
    --resume_lora0=models/alignment1/3mixed/300/best_checkpoint.pt \
    --resume_linear=models/alignment2/3mixed/1800-500/best_checkpoint.pt \
    --projector=linear \
    --linear_output_dim=4096 \
    --train_dir="datasets/finetuning/5/$dataset_name_s/train" \
    --train_dir_t="datasets/finetuning/5/$dataset_name_t/test" \
    --test_dir="datasets/finetuning/5/$dataset_name_t/test" \
    --save_dir="models/finetuning/${dataset_name_s}-${dataset_name_t}-5" \
    --llm="Qwen3-VL-8B-Instruct" \
    --nodistributed \
    --log_freq=4 \
    --batch_size=1 \
    --batch_size_t=1 \
    --accumulation_steps=4 \
    --epochs=10 