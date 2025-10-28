import bm25s
import Stemmer  # optional: for stemming
from pathlib import Path
from base_retriever import BaseRetriever
from utils import load_corpus

class BM25Retriver(BaseRetriver):
    def __init__(self, tokenizer, db):
        super(BM25Retriver, self).__init__(tokenizer=tokenizer, db=db)
        self.retriever = bm25s.BM25()
        self.init()

    def retrieve(self, input_ids, top_k=1):
        results, scores = retriever.retrieve(query_tokens, k=top_k)
        return results

    def init(self):
        db_path = Path(self.db)
        if db_path.exists()：
            self.retriever = bm25s.BM25.load(f"{self.db}_{self.tokenizer}_index", load_corpus=True)
        else:
            corpus_tokens = load_corpus(self.db, self.tokenizer)
            self.retriever.index(corpus_tokens)
            self.retriever.save(f"{self.db}_{self.tokenizer}_index")
        
        