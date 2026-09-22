"""
db.py
-----
Logs every question/answer pair (plus which chunks were used and how
long it took) into PostgreSQL, for later review or analytics.
"""

import psycopg2
from psycopg2.extras import Json
from datetime import datetime
from typing import List, Dict

from config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD,
)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources JSONB,
    latency_seconds REAL
);
"""


def get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def init_db():
    """Create the logging table if it doesn't exist yet. Safe to call
    every time the app starts."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def log_query(query: str, answer: str, sources: List[Dict], latency_seconds: float):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_logs (created_at, query, answer, sources, latency_seconds)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (datetime.utcnow(), query, answer, Json(sources), latency_seconds),
            )
        conn.commit()
    finally:
        conn.close()


def fetch_recent_logs(limit: int = 20):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, query, answer, latency_seconds
                FROM query_logs
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()
    finally:
        conn.close()
