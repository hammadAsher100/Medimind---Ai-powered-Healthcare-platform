from rag.embeddings.cohere_embedder import CohereEmbedder
from rag.vector_store.qdrant_store import QdrantStore


class Retriever:
    def __init__(self):
        self.embedder = CohereEmbedder()
        self.store = QdrantStore()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self.embedder.embed_texts([query], input_type="search_query")[0]
        return self.store.search(embedding, top_k=top_k)
