from torch import Tensor
from embedder import DenseEmbedder
# from retriver import Retriever
# from reranker import Reranker

K = 20

def retrieve(
    index: str,
    k: int,
    text: list[str]
):
    embedder = DenseEmbedder()
    embeddings, batch_dict = embedder.embedding(text)
    return embeddings

    # retriever = Retriever(index=index)
    # related_corpus = retriever.retrieve(K=K, embeddings=embeddings)

    # reranker = Reranker()
    # reranked_corpus = reranker.rerank(k=k, corpus=related_corpus)

    # return reranked_corpus    
    
if __name__ == '__main__':
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'
    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    input_texts = queries + documents
    
    import torch.nn.functional as F
    embeddings = retrieve(None, None, input_texts)
    print(embeddings.shape)
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    # scores = (embeddings[:2] @ embeddings[2:].T)
    # print(scores.tolist())