#!/usr/bin/env python3
"""
cluster_personas.py — Cluster approved campaign embeddings by persona
using K-Means and output a summary report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rag import chroma_client as cc
from rag import embedder


def cluster_personas(n_clusters: int = 4) -> None:
    """
    Retrieve all approved campaign embeddings, run K-Means clustering,
    and print a DataFrame summary with cluster labels and top terms.
    """
    print(f"Fetching approved_campaigns collection…")

    col = cc.get_collection(cc.APPROVED_CAMPAIGNS)
    count = col.count()
    if count == 0:
        print("No documents in approved_campaigns. Run `make index` first.")
        return

    result = col.get(include=["documents", "metadatas", "embeddings"])
    documents: list[str] = result.get("documents") or []
    metadatas: list[dict] = result.get("metadatas") or []
    raw_embeddings = result.get("embeddings") or []

    if not raw_embeddings:
        print("No embeddings found. Re-index with `make index`.")
        return

    X = np.array(raw_embeddings, dtype=float)
    print(f"Loaded {len(documents)} documents. Running K-Means with {n_clusters} clusters…")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    actual_clusters = min(n_clusters, len(documents))
    km = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    df = pd.DataFrame({
        "cluster": labels,
        "pca_x": X_2d[:, 0],
        "pca_y": X_2d[:, 1],
        "channel": [m.get("channel", "unknown") for m in metadatas],
        "brand": [m.get("brand", "unknown") for m in metadatas],
        "persona": [m.get("persona", "unknown") for m in metadatas],
        "document_preview": [d[:80] + "…" if len(d) > 80 else d for d in documents],
    })

    print("\n=== Cluster Summary ===")
    summary = df.groupby("cluster").agg(
        count=("cluster", "size"),
        channels=("channel", lambda x: ", ".join(sorted(set(x)))),
        brands=("brand", lambda x: ", ".join(sorted(set(x)))),
        personas=("persona", lambda x: ", ".join(sorted(set(x)))),
    )
    print(summary.to_string())

    print("\n=== Sample Documents per Cluster ===")
    for cluster_id in sorted(df["cluster"].unique()):
        print(f"\n--- Cluster {cluster_id} ---")
        sample = df[df["cluster"] == cluster_id].head(2)
        for _, row in sample.iterrows():
            print(f"  [{row['channel']}] {row['document_preview']}")

    out_path = Path("data/cluster_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = df.to_dict(orient="records")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Full report saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster persona embeddings")
    parser.add_argument("--clusters", type=int, default=4, help="Number of K-Means clusters")
    args = parser.parse_args()
    cluster_personas(n_clusters=args.clusters)
