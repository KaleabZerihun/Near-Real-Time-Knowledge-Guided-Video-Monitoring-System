import sqlite3
from pathlib import Path

def init_db(database_url: str) -> None:
    # Expect: sqlite:///./backend/data/app.db
    if not database_url.startswith("sqlite:///"):
        raise ValueError("DATABASE_URL must be sqlite:///... for this setup")

    # Resolve database path
    db_path = Path(database_url.replace("sqlite:///", "", 1)).resolve()

    # Ensure parent folder exists (e.g., backend/data/)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # This file lives at: backend/src/db/init_db.py
    # parents[2] => backend/
    backend_dir = Path(__file__).resolve().parents[2]
    schema_path = backend_dir / "db" / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"Missing schema.sql at: {schema_path}")

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(schema_path.read_text(encoding="utf-8"))
