"""
ClaimLens — Database Management Module
Handles SQLite persistence for analysis history, dashboard metrics, and records.
All operations use parameterized queries and safe context-managed connections.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "claimlens.db")


@contextmanager
def get_db(db_path: Optional[str] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager ensuring connection is always closed cleanly on Windows."""
    target_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize SQLite database tables with appropriate schema and indexes."""
    target_path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    with get_db(target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_text TEXT NOT NULL,
                model_prediction TEXT NOT NULL,
                model_score REAL,
                factcheck_status TEXT NOT NULL,
                factcheck_publisher TEXT,
                factcheck_url TEXT,
                relevance_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analyses_created_at
            ON analyses(created_at DESC)
            """
        )
        conn.commit()


def save_analysis(
    claim_text: str,
    model_prediction: str,
    model_score: Optional[float] = None,
    factcheck_status: str = "UNVERIFIED",
    factcheck_publisher: Optional[str] = None,
    factcheck_url: Optional[str] = None,
    relevance_summary: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """
    Save an analysis record into SQLite using parameterized queries.
    Returns the newly inserted record ID.
    """
    init_db(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO analyses (
                claim_text,
                model_prediction,
                model_score,
                factcheck_status,
                factcheck_publisher,
                factcheck_url,
                relevance_summary,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim_text.strip(),
                model_prediction,
                model_score,
                factcheck_status,
                factcheck_publisher,
                factcheck_url,
                relevance_summary,
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0


def get_history(
    limit: int = 50,
    search_query: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve historical analysis records ordered by newest first.
    Optionally filter by keyword in claim_text or model_prediction.
    """
    init_db(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        if search_query and search_query.strip():
            param = f"%{search_query.strip()}%"
            cursor.execute(
                """
                SELECT * FROM analyses
                WHERE claim_text LIKE ? OR model_prediction LIKE ? OR factcheck_status LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (param, param, param, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM analyses
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_analysis(analysis_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a single analysis record by ID."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def delete_analysis(analysis_id: int, db_path: Optional[str] = None) -> bool:
    """Delete a single analysis by ID. Returns True if a record was removed."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_history(db_path: Optional[str] = None) -> int:
    """Clear all records from the analyses table. Returns count of deleted rows."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analyses")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM analyses")
        conn.commit()
        return count


def get_dashboard_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute actual aggregate metrics from SQLite for dashboard presentation.
    Never invents statistics or returns ungrounded counts.
    """
    init_db(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM analyses")
        total = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM analyses WHERE model_prediction = 'LIKELY REAL'"
        )
        likely_real = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM analyses WHERE model_prediction = 'LIKELY FAKE'"
        )
        likely_fake = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM analyses WHERE factcheck_status = 'UNVERIFIED'"
        )
        unverified = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM analyses WHERE factcheck_status LIKE '%FOUND%'"
        )
        factchecks_found = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT id, claim_text, model_prediction, model_score, factcheck_status, created_at
            FROM analyses
            ORDER BY created_at DESC
            LIMIT 5
            """
        )
        recent = [dict(row) for row in cursor.fetchall()]

        return {
            "total_analyses": total,
            "likely_real_count": likely_real,
            "likely_fake_count": likely_fake,
            "unverified_count": unverified,
            "factchecks_found_count": factchecks_found,
            "recent_analyses": recent,
        }
