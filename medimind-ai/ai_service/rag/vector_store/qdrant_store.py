import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantStore:
    def __init__(self):
        self.collection = os.environ.get("QDRANT_COLLECTION", "medimind_knowledge")
        self.client = QdrantClient(
            host=os.environ.get("QDRANT_HOST", "qdrant"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
        )

    def create_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(collection.name == self.collection for collection in collections):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )

    def upsert_documents(self, chunks: list[dict]) -> int:
        points = []
        for chunk in chunks:
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk["embedding"],
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk.get("metadata", {}),
                        "title": chunk.get("title", ""),
                        "source": chunk.get("source", ""),
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        hits = self.client.search(collection_name=self.collection, query_vector=query_embedding, limit=top_k)
        return [{"score": hit.score, **hit.payload} for hit in hits]
