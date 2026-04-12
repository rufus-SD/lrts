from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path.home() / ".lrts"

DB_PATH = os.getenv("LRTS_DB_PATH", str(BASE_DIR / "lrts.db"))
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

DEFAULT_CONCURRENCY = int(os.getenv("LRTS_CONCURRENCY", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("LRTS_TEMPERATURE", "0"))
DEFAULT_SEED = int(os.getenv("LRTS_SEED", "42"))
SIMILARITY_THRESHOLD = float(os.getenv("LRTS_SIMILARITY_THRESHOLD", "0.85"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
