from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("SQLITE_DB_PATH", "./data/test_craftai.db")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test_chroma")
os.environ.setdefault("MLFLOW_TRACKING_URI", "./data/test_mlruns")
os.environ.setdefault("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
