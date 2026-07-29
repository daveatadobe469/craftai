#!/usr/bin/env python3
"""
reindex_chroma.py — Clean re-index of all ChromaDB collections.

Drops the three collections (brand_guidelines, approved_campaigns,
social_content) and rebuilds them from the canonical seed sources with fresh
embeddings. Use this after switching EMBEDDING_MODEL, or any time the vector
store has drifted (duplicates, ad-hoc inserts, mixed embedding dimensions).

Usage:
    python scripts/reindex_chroma.py            # reindex (preserves human-curated docs)
    python scripts/reindex_chroma.py --force    # skip the confirmation prompt
    python scripts/reindex_chroma.py --drop-curated
                                                # also discard human-curated (approved) docs

By default, documents the curator wrote back to approved_campaigns (identified
by a brief_id / doc_base_id in their metadata) are preserved and re-embedded so
the human-approved feedback loop is not lost on a re-index.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from rag import chroma_client as cc
from rag import embedder
from scripts.init_chroma import seed_campaigns, seed_social_content
from scripts.generate_guidelines import seed_guidelines

_COLLECTIONS = (cc.BRAND_GUIDELINES, cc.APPROVED_CAMPAIGNS, cc.SOCIAL_CONTENT)


def _snapshot_curated() -> list[dict[str, Any]]:
    """Return human-curated approved_campaigns docs (those with a brief_id)."""
    try:
        col = cc.get_collection(cc.APPROVED_CAMPAIGNS)
        res = col.get(include=["documents", "metadatas"])
    except Exception as exc:  # collection may not exist yet
        print(f"  (no existing approved_campaigns to snapshot: {exc})")
        return []

    curated: list[dict[str, Any]] = []
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    for doc_id, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        if meta.get("brief_id") or meta.get("doc_base_id"):
            curated.append({"id": doc_id, "document": doc, "metadata": meta})
    return curated


def _drop_collections() -> None:
    client = cc.get_client()
    for name in _COLLECTIONS:
        try:
            client.delete_collection(name)
            print(f"  ✗ dropped {name}")
        except Exception as exc:
            # Not fatal — a missing collection is fine on a fresh store.
            print(f"  · {name} not dropped ({exc})")
    cc.init_collections()
    print("  ✓ collections recreated (empty)")


def _restore_curated(curated: list[dict[str, Any]]) -> None:
    if not curated:
        return
    print(f"Restoring {len(curated)} human-curated document(s) with fresh embeddings…")
    texts = [c["document"] for c in curated]
    embeddings = embedder.encode(texts)
    cc.upsert_documents(
        collection_name=cc.APPROVED_CAMPAIGNS,
        ids=[c["id"] for c in curated],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in curated],
    )
    print(f"  ✓ restored {len(curated)} curated document(s)")


def reindex(drop_curated: bool = False) -> None:
    print(f"ChromaDB persist dir: {settings.CHROMA_PERSIST_DIR}")
    print(f"Embedding model:      {settings.EMBEDDING_MODEL}\n")

    curated = [] if drop_curated else _snapshot_curated()
    if curated:
        print(f"Preserving {len(curated)} human-curated approved doc(s) across reindex.\n")

    print("Dropping and recreating collections…")
    _drop_collections()

    print("\nReseeding from canonical sources…")
    seed_guidelines()
    seed_campaigns()
    seed_social_content()

    _restore_curated(curated)

    print("\nFinal collection counts:")
    for name in _COLLECTIONS:
        print(f"  {name}: {cc.collection_count(name)}")
    print("\n✅ Clean re-index complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean re-index of ChromaDB collections.")
    parser.add_argument("--force", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument(
        "--drop-curated",
        action="store_true",
        help="Also discard human-curated (approved) documents instead of preserving them.",
    )
    args = parser.parse_args()

    if not args.force:
        print(
            "This will DROP and rebuild all ChromaDB collections at "
            f"{settings.CHROMA_PERSIST_DIR}."
        )
        if args.drop_curated:
            print("Human-curated approved documents WILL be discarded (--drop-curated).")
        else:
            print("Human-curated approved documents will be preserved and re-embedded.")
        reply = input("Continue? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return

    reindex(drop_curated=args.drop_curated)


if __name__ == "__main__":
    main()
