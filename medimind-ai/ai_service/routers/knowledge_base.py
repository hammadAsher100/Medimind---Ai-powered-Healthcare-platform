from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from rag.chunking.text_chunker import chunk_text
from rag.embeddings.cohere_embedder import CohereEmbedder
from rag.loaders.document_loader import load_pdf

router = APIRouter(tags=["knowledge-base"])


@router.post("/index-document")
async def index_document(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
    source: str = Form(""),
    document_id: int | None = Form(None),
):
    try:
        content = await file.read()
        loaded = load_pdf(content, {"document_id": document_id, "filename": file.filename})
        chunks = chunk_text(loaded["text"])
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks could be created from the document.")
        embeddings = CohereEmbedder().embed_texts(chunks, input_type="search_document")
        points = [
            {
                "text": chunk,
                "embedding": embedding,
                "metadata": loaded["metadata"],
                "title": title or file.filename,
                "source": source,
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        indexed = request.app.state.qdrant_store.upsert_documents(points)
        return {"indexed_chunks": indexed, "title": title or file.filename, "source": source}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
