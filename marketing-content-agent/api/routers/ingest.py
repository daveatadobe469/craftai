from __future__ import annotations

import asyncio
import io
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from db.sqlite import write_audit
from rag import chroma_client as cc
from rag import embedder

router = APIRouter()

_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB hard limit

_ALLOWED_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/pdf",
    "application/json",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_COLLECTION_CHOICES = {
    "brand_guidelines": cc.BRAND_GUIDELINES,
    "approved_campaigns": cc.APPROVED_CAMPAIGNS,
    "social_content": cc.SOCIAL_CONTENT,
}


class IngestResponse(BaseModel):
    doc_id: str
    collection: str
    filename: str
    chunks: int
    message: str


class IngestListResponse(BaseModel):
    collection: str
    total_documents: int
    sample: list[dict]


def _extract_text(content: bytes, filename: str) -> str:
    """
    Extract plain text from an uploaded file.
    Supports: .txt, .md, .csv, .json, .pdf, .docx
    Falls back to UTF-8 decode with error replacement for unknown types.
    """
    fname = filename.lower()

    if fname.endswith(".pdf"):
        try:
            import pdfminer.high_level as pdfminer
            return pdfminer.extract_text(io.BytesIO(content))
        except ImportError:
            try:
                import pypdf

                reader = pypdf.PdfReader(io.BytesIO(content))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages)
            except ImportError:
                return content.decode("utf-8", errors="replace")

    if fname.endswith(".docx"):
        try:
            import docx

            doc = docx.Document(io.BytesIO(content))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            return content.decode("utf-8", errors="replace")

    return content.decode("utf-8", errors="replace")


def _chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    """
    Chunk text using LangChain's RecursiveCharacterTextSplitter.
    Returns a list of non-empty string chunks.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_document(
    file: Annotated[UploadFile, File(description="Document to embed (PDF, DOCX, TXT, MD, CSV, JSON)")],
    collection: Annotated[str, Form(description="Target ChromaDB collection")] = "brand_guidelines",
    brand: Annotated[str, Form(description="Brand tag for metadata")] = "",
    channel: Annotated[str, Form(description="Channel tag for metadata")] = "",
    chunk_size: Annotated[int, Form(description="Characters per chunk (default 512)")] = 512,
    chunk_overlap: Annotated[int, Form(description="Overlap between chunks (default 64)")] = 64,
) -> IngestResponse:
    """
    Upload a document, extract text, chunk it, embed each chunk,
    and upsert the embeddings into the chosen ChromaDB collection.
    The embedded chunks will be used by the Agentic RAG retrieval pipeline.
    """
    if collection not in _COLLECTION_CHOICES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid collection '{collection}'. Choose from: {', '.join(_COLLECTION_CHOICES)}",
        )

    if chunk_size < 64 or chunk_size > 4096:
        raise HTTPException(status_code=422, detail="chunk_size must be between 64 and 4096.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="chunk_overlap must be >= 0 and < chunk_size.")

    raw_content = await file.read()
    if len(raw_content) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {_MAX_FILE_BYTES // 1024 // 1024} MB.",
        )
    if len(raw_content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    loop = asyncio.get_event_loop()
    filename = file.filename or "upload.txt"

    text = await loop.run_in_executor(None, _extract_text, raw_content, filename)
    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract any text from the uploaded file.")

    chunks = await loop.run_in_executor(None, _chunk_text, text, chunk_size, chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=422, detail="Text extracted but produced zero chunks.")

    doc_base_id = str(uuid.uuid4())
    ids = [f"{doc_base_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_filename": filename,
            "doc_base_id": doc_base_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "collection": collection,
            "brand": brand.strip() or "unspecified",
            "channel": channel.strip() or "all",
            "upload_type": "user_upload",
        }
        for i in range(len(chunks))
    ]

    embeddings = await loop.run_in_executor(None, embedder.encode, chunks)

    collection_name = _COLLECTION_CHOICES[collection]
    await loop.run_in_executor(
        None,
        lambda: cc.upsert_documents(
            collection_name=collection_name,
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        ),
    )

    write_audit(
        brief_id=doc_base_id,
        event_type="document_ingested",
        event_data={
            "filename": filename,
            "collection": collection,
            "chunks": len(chunks),
            "brand": brand or "unspecified",
            "channel": channel or "all",
            "bytes": len(raw_content),
        },
        actor="user_upload",
    )

    return IngestResponse(
        doc_id=doc_base_id,
        collection=collection,
        filename=filename,
        chunks=len(chunks),
        message=(
            f"Successfully embedded {len(chunks)} chunk(s) from '{filename}' "
            f"into '{collection}'. They are now available to the Agentic RAG pipeline."
        ),
    )


@router.get("/ingest/{collection}", response_model=IngestListResponse)
async def list_collection(collection: str) -> IngestListResponse:
    """
    Return the document count and a sample of documents from the named collection.
    Useful for verifying that ingestion succeeded.
    """
    if collection not in _COLLECTION_CHOICES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid collection. Choose from: {', '.join(_COLLECTION_CHOICES)}",
        )

    collection_name = _COLLECTION_CHOICES[collection]
    loop = asyncio.get_event_loop()

    count = await loop.run_in_executor(None, cc.collection_count, collection_name)

    sample: list[dict] = []
    if count > 0:
        col = cc.get_collection(collection_name)
        raw = await loop.run_in_executor(
            None,
            lambda: col.get(
                limit=5,
                include=["documents", "metadatas"],
            ),
        )
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        for doc, meta in zip(docs, metas):
            sample.append({
                "preview": doc[:200] + ("…" if len(doc) > 200 else ""),
                "metadata": meta or {},
            })

    return IngestListResponse(
        collection=collection,
        total_documents=count,
        sample=sample,
    )


@router.delete("/ingest/{collection}/{doc_id}", status_code=200)
async def delete_document(collection: str, doc_id: str) -> dict:
    """
    Delete all chunks belonging to a doc_base_id from the collection.
    """
    if collection not in _COLLECTION_CHOICES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid collection. Choose from: {', '.join(_COLLECTION_CHOICES)}",
        )

    collection_name = _COLLECTION_CHOICES[collection]
    loop = asyncio.get_event_loop()
    col = cc.get_collection(collection_name)

    result = await loop.run_in_executor(
        None,
        lambda: col.get(where={"doc_base_id": doc_id}, include=["documents"]),
    )
    chunk_ids = result.get("ids") or []

    if not chunk_ids:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for doc_id '{doc_id}' in '{collection}'.",
        )

    await loop.run_in_executor(None, lambda: col.delete(ids=chunk_ids))

    return {
        "doc_id": doc_id,
        "collection": collection,
        "chunks_deleted": len(chunk_ids),
        "message": f"Deleted {len(chunk_ids)} chunk(s) from '{collection}'.",
    }
