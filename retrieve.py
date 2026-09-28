"""Chroma similarity search over the active scenario's uploaded document(s)."""

from config import TOP_K


def retrieve_chunks(query: str, collection, embedder, top_k: int = TOP_K) -> tuple[list[dict], list[float]]:
    """Embed the query and return (top_k most relevant chunks with metadata, query embedding)."""
    query_embedding = embedder.encode([query]).tolist()[0]

    if collection.count() == 0:
        return [], query_embedding

    results = collection.query(query_embeddings=[query_embedding], n_results=min(top_k, collection.count()))

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    for text, meta, distance in zip(documents, metadatas, distances):
        chunks.append({
            "text": text,
            "filename": meta["filename"],
            "page": meta["page"],
            "distance": distance,
        })
    return chunks, query_embedding
