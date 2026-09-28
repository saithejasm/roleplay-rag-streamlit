"""PDF -> pages -> chunks -> embeddings -> Chroma upsert."""

from pypdf import PdfReader
import io

from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_pages(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract text per page. Returns [{"filename", "page" (1-indexed), "text"}, ...]."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"filename": filename, "page": i + 1, "text": text})
    return pages


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding window over paragraphs, falling back to a raw character window
    when a single paragraph exceeds chunk_size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            # Paragraph itself is too long: raw sliding window with overlap.
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = end - overlap
            current = ""

    if current:
        chunks.append(current)

    # Apply overlap between paragraph-based chunks by prefixing the tail of the
    # previous chunk, so context isn't lost at a boundary.
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            tail = chunks[i - 1][-overlap:]
            chunk = tail + "\n\n" + chunk
        overlapped.append(chunk)
    return overlapped


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk each page's text independently, so a chunk never crosses a page boundary."""
    chunks = []
    for page in pages:
        pieces = _split_text(page["text"], chunk_size, overlap)
        for idx, piece in enumerate(pieces):
            chunks.append({
                "filename": page["filename"],
                "page": page["page"],
                "chunk_index": idx,
                "text": piece,
            })
    return chunks


def embed_chunks(chunks: list[dict], embedder) -> list[list[float]]:
    texts = [c["text"] for c in chunks]
    return embedder.encode(texts).tolist()


def extract_txt(file_bytes: bytes, filename: str) -> list[dict]:
    """Read a plain-text file as a single page. Returns [{"filename", "page": 1, "text"}] or []."""
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        return []
    return [{"filename": filename, "page": 1, "text": text}]


def _upsert_chunks(chunks: list[dict], collection, embedder) -> int:
    if not chunks:
        return 0
    embeddings = embed_chunks(chunks, embedder)
    ids = [f"{c['filename']}::p{c['page']}::c{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"filename": c["filename"], "page": c["page"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


def ingest_files(uploaded_files, collection, embedder) -> int:
    """Ingest a list of Streamlit UploadedFile objects into the given Chroma collection.
    Returns the number of chunks added."""
    total_chunks = 0
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".txt"):
            pages = extract_txt(file_bytes, uploaded_file.name)
        else:
            pages = extract_pages(file_bytes, uploaded_file.name)
        chunks = chunk_pages(pages)
        total_chunks += _upsert_chunks(chunks, collection, embedder)
    return total_chunks


def ingest_text(text: str, filename: str, collection, embedder) -> int:
    """Ingest freeform typed/pasted text (e.g. a resume or case details typed directly
    instead of uploaded as a PDF) the same way a PDF page would be. Replaces any chunks
    previously ingested under the same filename first, so edits don't leave stale chunks
    behind when the new text is shorter than what it replaces."""
    text = text.strip()
    try:
        collection.delete(where={"filename": filename})
    except Exception:
        pass
    if not text:
        return 0
    pages = [{"filename": filename, "page": 1, "text": text}]
    chunks = chunk_pages(pages)
    return _upsert_chunks(chunks, collection, embedder)
