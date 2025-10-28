class BaseRetriever:
    def __init__(self, tokenizer, db):
        self.tokenzier = tokenizer
        self.db = db

    def retrieve(self, input_ids, top_k):
        raise NotImplementedError