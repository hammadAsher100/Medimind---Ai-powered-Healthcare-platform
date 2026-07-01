import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# In-memory conversation store as ultimate fallback
_memory_store: dict[int, list[dict]] = {}

# SQLite fallback path (same directory as FastAPI service)
_SQLITE_PATH = Path(__file__).resolve().parent.parent / "conversation_memory.db"


def _use_postgres() -> bool:
    """Check if Postgres is configured and available."""
    host = os.environ.get("DB_HOST") or os.environ.get("POSTGRES_HOST", "")
    return bool(host) and host not in ("", "localhost_disabled")


def _dsn() -> str:
    return (
        f"dbname={os.environ.get('POSTGRES_DB', os.environ.get('DB_NAME', 'medimind'))} "
        f"user={os.environ.get('POSTGRES_USER', os.environ.get('DB_USER', 'medimind'))} "
        f"password={os.environ.get('POSTGRES_PASSWORD', os.environ.get('DB_PASSWORD', 'medimind_password'))} "
        f"host={os.environ.get('POSTGRES_HOST', os.environ.get('DB_HOST', 'postgres'))} "
        f"port={os.environ.get('POSTGRES_PORT', os.environ.get('DB_PORT', '5432'))}"
    )


@contextmanager
def _pg_connection():
    import psycopg2
    conn = psycopg2.connect(_dsn())
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def _sqlite_connection():
    conn = sqlite3.connect(str(_SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_sqlite_table():
    with _sqlite_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def _ensure_pg_table():
    with _pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id BIGSERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)


def get_recent_memory(user_id: int, limit: int = 10) -> list[dict]:
    """Get recent conversation memory for a user."""
    # Try Postgres first
    if _use_postgres():
        try:
            from psycopg2.extras import RealDictCursor
            _ensure_pg_table()
            with _pg_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT role, content, metadata, created_at FROM conversation_memory WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
                        (user_id, limit),
                    )
                    return list(reversed(cur.fetchall()))
        except Exception as exc:
            logger.debug("Postgres memory fetch failed: %s", exc)

    # Try SQLite fallback
    try:
        _ensure_sqlite_table()
        with _sqlite_connection() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata, created_at FROM conversation_memory WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
            result = []
            for row in reversed(rows):
                meta = row["metadata"]
                try:
                    meta = json.loads(meta) if isinstance(meta, str) else meta
                except (json.JSONDecodeError, TypeError):
                    meta = {}
                result.append({"role": row["role"], "content": row["content"], "metadata": meta, "created_at": row["created_at"]})
            return result
    except Exception as exc:
        logger.debug("SQLite memory fetch failed: %s", exc)

    # In-memory fallback
    return _memory_store.get(user_id, [])[-limit:]


def save_conversation_turn(user_id: int, role: str, content: str, metadata: dict | None = None) -> None:
    """Save a conversation turn to persistent storage."""
    meta = metadata or {}

    # Try Postgres first
    if _use_postgres():
        try:
            from psycopg2.extras import Json
            _ensure_pg_table()
            with _pg_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO conversation_memory (user_id, role, content, metadata) VALUES (%s, %s, %s, %s)",
                        (user_id, role, content, Json(meta)),
                    )
            return
        except Exception as exc:
            logger.debug("Postgres memory save failed: %s", exc)

    # Try SQLite fallback
    try:
        _ensure_sqlite_table()
        with _sqlite_connection() as conn:
            conn.execute(
                "INSERT INTO conversation_memory (user_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                (user_id, role, content, json.dumps(meta)),
            )
        return
    except Exception as exc:
        logger.debug("SQLite memory save failed: %s", exc)

    # In-memory fallback
    _memory_store.setdefault(user_id, []).append({"role": role, "content": content, "metadata": meta})


def log_timeline_event(user_id: int, event_type: str, title: str, description: str, metadata: dict | None = None) -> None:
    """Log a timeline event. Only works with Postgres (Django's database)."""
    if not _use_postgres():
        return
    try:
        from psycopg2.extras import Json
        with _pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO timeline_timelineevent (user_id, event_type, title, description, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    """,
                    (user_id, event_type, title, description, Json(metadata or {})),
                )
    except Exception as exc:
        logger.debug("Timeline event log failed: %s", exc)


def summarize_if_needed(memory: list[dict]) -> list[dict]:
    """Summarize conversation memory if it exceeds word limit."""
    total_words = sum(len(item.get("content", "").split()) for item in memory)
    if total_words <= 2000:
        return memory
    preserved = memory[-6:]
    summary = "Earlier conversation included general health questions and context used for continuity."
    return [{"role": "system", "content": summary, "metadata": {"summary": True}}] + preserved
