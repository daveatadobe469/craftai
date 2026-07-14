from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path

from config import settings
from db.sqlite import write_audit
from rag import chroma_client as cc
from rag import embedder
from rag.text_extraction import chunk_text, extract_text

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".md"}
_IGNORED_FILENAMES = {"readme.md", "readme.txt", ".gitkeep"}

_MANIFEST_PATH = Path(settings.SQLITE_DB_PATH).parent / "brand_guideline_manifest.json"


def _load_manifest() -> dict[str, str]:
    if not _MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(manifest: dict[str, str]) -> None:
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ingest_brand_guideline_folder(folder: str | Path | None = None) -> dict[str, int]:
    """
    Scan `folder` (default: settings.BRAND_GUIDELINE_FOLDER) for supported
    documents and auto-embed any new or changed ones into the
    brand_guidelines ChromaDB collection. Tracks a filename -> content-hash
    manifest so unchanged files are skipped on subsequent calls (e.g. every
    API restart / --reload).
    """
    folder_path = Path(folder or settings.BRAND_GUIDELINE_FOLDER)
    summary = {"ingested": 0, "skipped": 0, "failed": 0}

    if not folder_path.is_dir():
        return summary

    manifest = _load_manifest()

    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.name.lower() in _IGNORED_FILENAMES:
            continue
        if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue

        try:
            raw_content = file_path.read_bytes()
            digest = hashlib.sha256(raw_content).hexdigest()

            if manifest.get(file_path.name) == digest:
                summary["skipped"] += 1
                continue

            text = extract_text(raw_content, file_path.name).strip()
            if not text:
                logger.warning("Skipping %s: no extractable text.", file_path.name)
                summary["failed"] += 1
                continue

            chunks = chunk_text(text)
            if not chunks:
                logger.warning("Skipping %s: produced zero chunks.", file_path.name)
                summary["failed"] += 1
                continue

            doc_base_id = str(uuid.uuid4())
            ids = [f"{doc_base_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source_filename": file_path.name,
                    "doc_base_id": doc_base_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "collection": "brand_guidelines",
                    "brand": "unspecified",
                    "channel": "all",
                    "upload_type": "auto_folder_ingest",
                }
                for i in range(len(chunks))
            ]

            embeddings = embedder.encode(chunks)
            cc.upsert_documents(
                collection_name=cc.BRAND_GUIDELINES,
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            write_audit(
                brief_id=doc_base_id,
                event_type="document_ingested",
                event_data={
                    "filename": file_path.name,
                    "collection": "brand_guidelines",
                    "chunks": len(chunks),
                    "bytes": len(raw_content),
                },
                actor="auto_folder_ingest",
            )

            manifest[file_path.name] = digest
            summary["ingested"] += 1
            logger.info("Auto-ingested %s (%d chunks).", file_path.name, len(chunks))
        except Exception:
            logger.exception("Failed to auto-ingest %s", file_path.name)
            summary["failed"] += 1

    _save_manifest(manifest)
    return summary
