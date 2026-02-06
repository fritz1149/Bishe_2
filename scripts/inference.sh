source /work/miniconda3/etc/profile.d/conda.sh
conda activate bishe
# cd Bishe_2

LOG_FILE="logs/inference/$(date +%s)_$$.txt"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[EXIT] $(date -Is) inference finished, exit_code=$rc, log=$LOG_FILE"' EXIT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dataset_name=qq;
k=500;
corpus_k=200;
initial_top_k=10;
max_iterations=3;

python -m z2.inference_test \
  --mode="complex" \
  --early_stop=20 \
  --resume_encoder=models/encoder/90000/best_checkpoint.pt \
  --resume_lora0_0=models/alignment1/3mixed/300/best_checkpoint.pt \
  --resume_linear_0=models/alignment2/3mixed/1800-500/best_checkpoint.pt \
  --resume_lora0_1=models/traffic_embedder/30000-2gpu/best_checkpoint.pt \
  --resume_linear_1=models/traffic_embedder/30000-2gpu/best_checkpoint.pt \
  --dataset_path=datasets/zero-shot/$k/$dataset_name/test \
  --batch_size=1 \
  --vector_index_dir=datasets/rag-emb-index/$corpus_k/index \
  --bm25_index_dir=datasets/rag-corpus-index/$corpus_k/index \
  --initial_top_k=$initial_top_k \
  --max_iterations=$max_iterations \
  --parallel_mode \
  --inference_dtype=bf16 \
  --output_dir=logs/inference_results