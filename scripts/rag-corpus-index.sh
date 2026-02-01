cd Bishe_2

LOG_FILE="logs/corpus_generate/corpus-index/$(date +%s)_$$.txt"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[EXIT] $(date -Is) rag-corpus-index finished, exit_code=$rc, log=$LOG_FILE"' EXIT

source /work/miniconda3/etc/profile.d/conda.sh
conda activate bishe

k=2000
python -m z2.RAG.retriever.BM25 \
    build_index \
    --corpus_path=datasets/rag-corpus-index/$k \
    --index_dir=datasets/rag-corpus-index/$k/index
