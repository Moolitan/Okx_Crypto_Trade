# data/okx/path.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # ✅ 这里改成 2

DATABASE_DIR = PROJECT_ROOT / "database" / "db"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

OKX_DUCKDB_PATH = DATABASE_DIR / "okx.duckdb"
