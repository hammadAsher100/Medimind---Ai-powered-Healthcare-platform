import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import Json, RealDictCursor


def _dsn() -> str:
    return (
        f"dbname={os.environ.get('POSTGRES_DB', 'medimind')} "
        f"user={os.environ.get('POSTGRES_USER', 'medimind')} "
        f"password={os.environ.get('POSTGRES_PASSWORD', 'medimind_password')} "
        f"host={os.environ.get('POSTGRES_HOST', 'postgres')} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')}"
    )


@contextmanager
def connection():
    conn = psycopg2.connect(_dsn())
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_memory_table() -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id BIGSERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )


def get_recent_memory(user_id: int, limit: int = 10) -> list[dict]:
    ensure_memory_table()
    with connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, metadata, created_at FROM conversation_memory WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
                (user_id, limit),
            )
            return list(reversed(cur.fetchall()))


def save_conversation_turn(user_id: int, role: str, content: str, metadata: dict | None = None) -> None:
    ensure_memory_table()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_memory (user_id, role, content, metadata) VALUES (%s, %s, %s, %s)",
                (user_id, role, content, Json(metadata or {})),
            )


def summarize_if_needed(memory: list[dict]) -> list[dict]:
    total_words = sum(len(item.get("content", "").split()) for item in memory)
    if total_words <= 2000:
        return memory
    preserved = memory[-6:]
    summary = "Earlier conversation included general health questions and context used for continuity."
    return [{"role": "system", "content": summary, "metadata": {"summary": True}}] + preserved
