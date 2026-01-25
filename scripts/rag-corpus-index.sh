source /opt/anaconda3/etc/profile.d/conda.sh
conda activate bishe
cd Bishe_2

LOG_FILE="logs/corpus_generate/corpus-index/$(date +%s)_$$.txt"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[EXIT] $(date -Is) rag-corpus-index finished, exit_code=$rc, log=$LOG_FILE"' EXIT

k=2000
python -m z2.corpus_generate \
    run_text_corpus_pipeline \
    --dataset_path=datasets/rag-corpus/$k/3mixed \
    --corpus_output_dir=datasets/rag-corpus-index/$k \
    --index_output_dir=datasets/rag-corpus-index/$k/index \
    --resume_log \
    --resume_encoder=models/encoder/90000/best_checkpoint.pt \
    --resume_lora0=models/alignment1/3mixed/300/best_checkpoint.pt \
    --resume_linear=models/alignment2/3mixed/1800-500/best_checkpoint.pt