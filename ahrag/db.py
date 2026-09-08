"""SQLite persistence for AHRAG.

Holds document metadata, chunks, demo users and role assignments, audit
records, and evaluation runs. Deliberately a thin repository over ``sqlite3``
rather than an ORM: the schema is small, the queries are explicit, and a
reviewer can read exactly what is and is not persisted -- which matters for the
privacy claims made about the audit log.

Connections are created per-``Database`` instance with ``check_same_thread=False``
so that FastAPI's threadpool and Streamlit can share one instance safely; all
writes go through a lock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from .models import AuditRecord, Chunk, DocType, DocumentMeta, User

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    source_uri      TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    owner           TEXT NOT NULL,
    acl_roles       TEXT NOT NULL,
    created_date    TEXT NOT NULL,
    effective_date  TEXT NOT NULL,
    version         TEXT NOT NULL,
    authority_score INTEGER NOT NULL,
    policy_family   TEXT,
    supersedes      TEXT,
    superseded_by   TEXT,
    checksum        TEXT,
    ingested_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    heading     TEXT,
    text        TEXT NOT NULL,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS roles (
    role        TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role    TEXT NOT NULL,
    PRIMARY KEY (user_id, role)
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id            TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    user_roles          TEXT NOT NULL,
    query_hash          TEXT NOT NULL,
    query_text          TEXT,
    route               TEXT NOT NULL,
    fallback_route      TEXT NOT NULL,
    router_version      TEXT NOT NULL,
    router_confidence   REAL NOT NULL,
    features            TEXT NOT NULL,
    candidate_utilities TEXT NOT NULL,
    timings_ms          TEXT NOT NULL,
    documents_considered TEXT NOT NULL,
    documents_used      TEXT NOT NULL,
    acl_pool_size       INTEGER NOT NULL,
    acl_withheld_count  INTEGER NOT NULL,
    citations           TEXT NOT NULL,
    abstained           INTEGER NOT NULL,
    abstention_reason   TEXT NOT NULL,
    conflicts_disclosed INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd  REAL NOT NULL,
    total_latency_s     REAL NOT NULL,
    generator           TEXT NOT NULL,
    verbose             INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp DESC);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    config_hash TEXT NOT NULL,
    notes       TEXT NOT NULL DEFAULT '',
    summary     TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id      TEXT NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    system      TEXT NOT NULL,
    query_id    TEXT NOT NULL,
    payload     TEXT NOT NULL,
    PRIMARY KEY (run_id, system, query_id)
);
"""


def _json(value: Any) -> str:
    """Serialise ``value`` to JSON with date/enum support."""

    def default(obj: Any) -> Any:
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if hasattr(obj, "value"):
            return obj.value
        return str(obj)

    return json.dumps(value, default=default, ensure_ascii=False)


class Database:
    """Repository over the AHRAG SQLite file."""

    def __init__(self, path: str | Path) -> None:
        """Open (creating if needed) the database at ``path``.

        Pass ``":memory:"`` for an ephemeral database, which the tests use.
        """
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    def reset_corpus(self) -> None:
        """Delete all documents and chunks, leaving users and audit intact."""
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM documents")
            self._conn.commit()

    # -- documents & chunks ------------------------------------------------

    def upsert_document(self, meta: DocumentMeta) -> None:
        """Insert or replace one document's metadata."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO documents (doc_id, title, source_uri, doc_type, owner,
                    acl_roles, created_date, effective_date, version, authority_score,
                    policy_family, supersedes, superseded_by, checksum, ingested_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title, source_uri=excluded.source_uri,
                    doc_type=excluded.doc_type, owner=excluded.owner,
                    acl_roles=excluded.acl_roles, created_date=excluded.created_date,
                    effective_date=excluded.effective_date, version=excluded.version,
                    authority_score=excluded.authority_score,
                    policy_family=excluded.policy_family,
                    supersedes=excluded.supersedes,
                    superseded_by=excluded.superseded_by,
                    checksum=excluded.checksum, ingested_at=excluded.ingested_at
                """,
                (
                    meta.doc_id,
                    meta.title,
                    meta.source_uri,
                    meta.doc_type.value,
                    meta.owner,
                    _json(meta.acl_roles),
                    meta.created_date.isoformat(),
                    meta.effective_date.isoformat(),
                    meta.version,
                    meta.authority_score,
                    meta.policy_family,
                    meta.supersedes,
                    meta.superseded_by,
                    meta.checksum,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            self._conn.commit()

    def set_superseded_by(self, doc_id: str, superseded_by: str | None) -> None:
        """Record that ``doc_id`` has been replaced by ``superseded_by``."""
        with self._lock:
            self._conn.execute(
                "UPDATE documents SET superseded_by=? WHERE doc_id=?",
                (superseded_by, doc_id),
            )
            self._conn.commit()

    def delete_document(self, doc_id: str) -> None:
        """Remove a document and its chunks."""
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self._conn.commit()

    def insert_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Insert chunk rows; returns the number written."""
        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.ordinal,
                c.heading,
                c.text,
                c.char_start,
                c.char_end,
            )
            for c in chunks
        ]
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                """INSERT OR REPLACE INTO chunks
                   (chunk_id, doc_id, ordinal, heading, text, char_start, char_end)
                   VALUES (?,?,?,?,?,?,?)""",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def get_documents(self) -> list[DocumentMeta]:
        """Return every document's metadata, ordered by title."""
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM documents ORDER BY title")
            return [_row_to_document(r) for r in cursor.fetchall()]

    def get_document(self, doc_id: str) -> DocumentMeta | None:
        """Return one document's metadata, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
        return _row_to_document(row) if row else None

    def get_chunks(self) -> list[Chunk]:
        """Return every chunk with parent governance metadata denormalised on."""
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.ordinal, c.heading, c.text,
                       c.char_start, c.char_end,
                       d.title, d.source_uri, d.doc_type, d.owner, d.acl_roles,
                       d.created_date, d.effective_date, d.version,
                       d.authority_score, d.policy_family, d.supersedes,
                       d.superseded_by
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                ORDER BY d.title, c.ordinal
                """
            )
            return [_row_to_chunk(r) for r in cursor.fetchall()]

    def count_chunks(self) -> int:
        """Return the number of chunks in the corpus."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # -- users & roles -----------------------------------------------------

    def upsert_user(self, user: User) -> None:
        """Insert or replace a demo user and their role assignments."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO users (user_id, display_name, description)
                   VALUES (?,?,?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     display_name=excluded.display_name,
                     description=excluded.description""",
                (user.user_id, user.display_name, user.description),
            )
            self._conn.execute("DELETE FROM user_roles WHERE user_id=?", (user.user_id,))
            self._conn.executemany(
                "INSERT OR IGNORE INTO user_roles (user_id, role) VALUES (?,?)",
                [(user.user_id, role) for role in user.roles],
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO roles (role, description) VALUES (?,'')",
                [(role,) for role in user.roles],
            )
            self._conn.commit()

    def get_users(self) -> list[User]:
        """Return all demo users with their roles."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id, display_name, description FROM users ORDER BY user_id"
            ).fetchall()
            users: list[User] = []
            for row in rows:
                roles = [
                    r[0]
                    for r in self._conn.execute(
                        "SELECT role FROM user_roles WHERE user_id=? ORDER BY role",
                        (row["user_id"],),
                    ).fetchall()
                ]
                users.append(
                    User(
                        user_id=row["user_id"],
                        display_name=row["display_name"],
                        description=row["description"],
                        roles=roles,
                    )
                )
        return users

    def get_user(self, user_id: str) -> User | None:
        """Return one demo user, or None when unknown."""
        for user in self.get_users():
            if user.user_id == user_id:
                return user
        return None

    def get_roles(self) -> list[str]:
        """Return every known role name."""
        with self._lock:
            return [r[0] for r in self._conn.execute("SELECT role FROM roles ORDER BY role")]

    # -- audit -------------------------------------------------------------

    def write_audit(self, record: AuditRecord) -> None:
        """Persist one audit record."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO audit_log VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.audit_id,
                    record.timestamp.isoformat(timespec="seconds"),
                    record.user_id,
                    _json(record.user_roles),
                    record.query_hash,
                    record.query_text,
                    record.route.value,
                    record.fallback_route.value,
                    record.router_version,
                    record.router_confidence,
                    _json(record.features),
                    _json(record.candidate_utilities),
                    _json(record.timings_ms),
                    _json(record.documents_considered),
                    _json(record.documents_used),
                    record.acl_pool_size,
                    record.acl_withheld_count,
                    _json(record.citations),
                    int(record.abstained),
                    record.abstention_reason.value,
                    record.conflicts_disclosed,
                    record.estimated_cost_usd,
                    record.total_latency_s,
                    record.generator,
                    int(record.verbose),
                ),
            )
            self._conn.commit()

    def get_audit_records(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent audit records as plain dicts for display."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for field in (
                "user_roles",
                "features",
                "candidate_utilities",
                "timings_ms",
                "documents_considered",
                "documents_used",
                "citations",
            ):
                try:
                    item[field] = json.loads(item[field])
                except (TypeError, json.JSONDecodeError):
                    item[field] = []
            item["abstained"] = bool(item["abstained"])
            item["verbose"] = bool(item["verbose"])
            out.append(item)
        return out

    def clear_audit(self) -> None:
        """Delete all audit records (demo convenience)."""
        with self._lock:
            self._conn.execute("DELETE FROM audit_log")
            self._conn.commit()

    # -- evaluation runs ---------------------------------------------------

    def start_eval_run(self, run_id: str, config_hash: str, notes: str = "") -> None:
        """Record the start of an evaluation run."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO eval_runs
                   (run_id, started_at, finished_at, config_hash, notes, summary)
                   VALUES (?,?,NULL,?,?,NULL)""",
                (run_id, datetime.now().isoformat(timespec="seconds"), config_hash, notes),
            )
            self._conn.commit()

    def finish_eval_run(self, run_id: str, summary: dict[str, Any]) -> None:
        """Record the completion and summary metrics of an evaluation run."""
        with self._lock:
            self._conn.execute(
                "UPDATE eval_runs SET finished_at=?, summary=? WHERE run_id=?",
                (datetime.now().isoformat(timespec="seconds"), _json(summary), run_id),
            )
            self._conn.commit()

    def write_eval_result(
        self, run_id: str, system: str, query_id: str, payload: dict[str, Any]
    ) -> None:
        """Persist one per-query evaluation result."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO eval_results VALUES (?,?,?,?)",
                (run_id, system, query_id, _json(payload)),
            )
            self._conn.commit()

    def get_eval_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent evaluation runs, newest first, with parsed summaries."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM eval_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            if item.get("summary"):
                try:
                    item["summary"] = json.loads(item["summary"])
                except json.JSONDecodeError:
                    item["summary"] = None
            out.append(item)
        return out


# ---------------------------------------------------------------------------
# Row mappers
# ---------------------------------------------------------------------------


def _row_to_document(row: sqlite3.Row) -> DocumentMeta:
    """Map a ``documents`` row to :class:`DocumentMeta`."""
    return DocumentMeta(
        doc_id=row["doc_id"],
        title=row["title"],
        source_uri=row["source_uri"],
        doc_type=DocType(row["doc_type"]),
        owner=row["owner"],
        acl_roles=json.loads(row["acl_roles"]),
        created_date=date.fromisoformat(row["created_date"]),
        effective_date=date.fromisoformat(row["effective_date"]),
        version=row["version"],
        authority_score=row["authority_score"],
        policy_family=row["policy_family"],
        supersedes=row["supersedes"],
        superseded_by=row["superseded_by"],
        checksum=row["checksum"],
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    """Map a joined ``chunks``/``documents`` row to :class:`Chunk`."""
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        ordinal=row["ordinal"],
        heading=row["heading"],
        text=row["text"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        title=row["title"],
        source_uri=row["source_uri"],
        doc_type=DocType(row["doc_type"]),
        owner=row["owner"],
        acl_roles=json.loads(row["acl_roles"]),
        created_date=date.fromisoformat(row["created_date"]),
        effective_date=date.fromisoformat(row["effective_date"]),
        version=row["version"],
        authority_score=row["authority_score"],
        policy_family=row["policy_family"],
        supersedes=row["supersedes"],
        superseded_by=row["superseded_by"],
    )
