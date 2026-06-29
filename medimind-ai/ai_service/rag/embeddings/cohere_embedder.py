import os

import cohere

MODEL = "embed-english-v3.0"


class CohereEmbedder:
    def __init__(self):
        api_key = (os.environ.get("COHERE_API_KEY") or "").strip()
        self.client = cohere.Client(api_key) if api_key else None

    def embed_texts(self, texts: list[str], input_type: str = "search_document") -> list[list[float]]:
        if not self.client:
            raise RuntimeError("COHERE_API_KEY is not configured.")
        vectors: list[list[float]] = []
        for index in range(0, len(texts), 96):
            batch = texts[index : index + 96]
            response = self.client.embed(texts=batch, model=MODEL, input_type=input_type)
            vectors.extend(response.embeddings)
        return vectors
