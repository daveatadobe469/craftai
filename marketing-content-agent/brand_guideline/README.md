Drop brand guideline documents in this folder (`.md` files only).

On every API startup, `rag/folder_ingest.py` scans this folder and automatically
chunks, embeds, and upserts any new or changed file into the `brand_guidelines`
ChromaDB collection — no upload button required.

A manifest of already-ingested files (by content hash) is kept at
`data/brand_guideline_manifest.json` so unchanged files are skipped on
subsequent restarts. Delete a file's entry there (or the whole manifest) to
force re-ingestion.

This README is ignored by the auto-ingest scan.
