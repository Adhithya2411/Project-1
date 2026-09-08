"""Ingestion service: manifest seeding and ad-hoc document upload."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..config import Settings
from ..db import Database
from ..models import Chunk, DocType, DocumentMeta, User
from .chunker import chunk_document, resolve_supersession
from .loaders import LoaderError, load_text

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"

# An uploaded document with no declared ACL gets this, never "everyone". A
# prototype that silently widens access on the unhappy path would undercut the
# governance claim it exists to demonstrate.
DEFAULT_UPLOAD_ACL = ["hr", "manager"]


@dataclass
class IngestionResult:
    """Summary of one ingestion pass."""

    documents: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "errors": self.errors,
            "doc_ids": self.doc_ids,
        }


class IngestionService:
    """Loads documents into SQLite and returns the chunks for indexing.

    The service does not own the search indexes; it returns chunks and the
    caller (``AHRAGEngine``) rebuilds the indexes. Keeping index construction
    out of ingestion means the same chunks can be indexed by different backends
    in the evaluation harness without re-parsing files.
    """

    def __init__(self, db: Database, settings: Settings) -> None:
        """Bind the service to a database and settings instance."""
        self.db = db
        self.settings = settings

    # -- seeding -----------------------------------------------------------

    def seed_from_manifest(
        self, manifest_path: str | Path | None = None, reset: bool = True
    ) -> IngestionResult:
        """Ingest the demo corpus described by ``manifest_path``.

        Args:
            manifest_path: YAML manifest. Defaults to the packaged seed manifest.
            reset: Drop existing documents and chunks first. Users and audit
                records are preserved.

        Returns:
            An :class:`IngestionResult`. Per-document failures are collected in
            ``errors`` rather than aborting the whole seed, so one unreadable
            file does not leave the corpus empty.

        Raises:
            FileNotFoundError: If the manifest itself is missing.
            ValueError: If the manifest is malformed.
        """
        path = Path(manifest_path) if manifest_path else SEED_DIR / "manifest.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Seed manifest not found: {path}")
        try:
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Seed manifest {path} is not valid YAML: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ValueError(f"Seed manifest {path} must contain a mapping.")

        if reset:
            self.db.reset_corpus()

        self._seed_users(manifest.get("users", []))

        base = path.parent
        metas: list[DocumentMeta] = []
        texts: dict[str, str] = {}
        result = IngestionResult()

        for entry in manifest.get("documents", []):
            try:
                meta, text = self._load_manifest_document(entry, base)
            except (LoaderError, ValueError, KeyError) as exc:
                message = f"{entry.get('doc_id', '<unknown>')}: {exc}"
                logger.warning("Skipping document during seed — %s", message)
                result.errors.append(message)
                continue
            metas.append(meta)
            texts[meta.doc_id] = text

        # Derive superseded_by from the declared supersedes links before
        # chunking, so retirement markers are baked into every chunk.
        metas = resolve_supersession(metas)

        for meta in metas:
            self.db.upsert_document(meta)
            chunks = chunk_document(
                texts[meta.doc_id],
                meta,
                target_chars=self.settings.chunk_target_chars,
                overlap_chars=self.settings.chunk_overlap_chars,
            )
            written = self.db.insert_chunks(chunks)
            result.documents += 1
            result.chunks += written
            result.doc_ids.append(meta.doc_id)

        logger.info(
            "Seeded corpus: %d documents, %d chunks, %d errors",
            result.documents,
            result.chunks,
            len(result.errors),
        )
        return result

    def _seed_users(self, entries: list[dict[str, Any]]) -> None:
        """Insert or update the demo users described in the manifest."""
        for entry in entries:
            try:
                user = User(
                    user_id=entry["user_id"],
                    display_name=entry.get("display_name", entry["user_id"]),
                    roles=list(entry.get("roles", [])),
                    description=entry.get("description", ""),
                )
            except KeyError as exc:
                logger.warning("Skipping malformed user entry %s: %s", entry, exc)
                continue
            self.db.upsert_user(user)

    def _load_manifest_document(
        self, entry: dict[str, Any], base: Path
    ) -> tuple[DocumentMeta, str]:
        """Build a :class:`DocumentMeta` and load its text from a manifest entry."""
        source = (base / entry["path"]).resolve()
        text = load_text(source)
        acl_roles = list(entry.get("acl_roles", []))
        if not acl_roles:
            raise ValueError(
                "manifest entry declares no acl_roles; refusing to ingest with "
                "implicit universal access"
            )
        meta = DocumentMeta(
            doc_id=entry["doc_id"],
            title=entry["title"],
            source_uri=str(source),
            doc_type=DocType(entry.get("doc_type", "other")),
            owner=entry.get("owner", "unassigned"),
            acl_roles=acl_roles,
            created_date=_as_date(entry["created_date"]),
            effective_date=_as_date(entry["effective_date"]),
            version=str(entry.get("version", "1.0")),
            authority_score=int(entry.get("authority_score", 3)),
            policy_family=entry.get("policy_family"),
            supersedes=entry.get("supersedes"),
            superseded_by=entry.get("superseded_by"),
            checksum=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        )
        return meta, text

    # -- ad-hoc upload -----------------------------------------------------

    def ingest_file(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        doc_type: DocType = DocType.OTHER,
        owner: str = "unassigned",
        acl_roles: list[str] | None = None,
        effective_date: date | None = None,
        version: str = "1.0",
        authority_score: int = 3,
        policy_family: str | None = None,
        supersedes: str | None = None,
    ) -> tuple[DocumentMeta, list[Chunk]]:
        """Ingest a single uploaded file.

        Args:
            path: File to ingest (``.txt``, ``.md``, ``.pdf``, ``.docx``).
            title: Display title; defaults to the file stem.
            doc_type: Document taxonomy entry.
            owner: Accountable owner.
            acl_roles: Roles permitted to read. Defaults to
                :data:`DEFAULT_UPLOAD_ACL` — deliberately restrictive.
            effective_date: Authoritative-from date; defaults to today.
            version: Version label.
            authority_score: 1..5 authority.
            policy_family: Groups versions of the same topic for conflict checks.
            supersedes: doc_id this document replaces.

        Returns:
            The stored metadata and the chunks written.

        Raises:
            LoaderError: If the file cannot be parsed.
            ValueError: If ``authority_score`` is out of range.
        """
        source = Path(path).resolve()
        text = load_text(source)
        if not 1 <= authority_score <= 5:
            raise ValueError("authority_score must be between 1 and 5")

        doc_id = f"doc-upload-{hashlib.sha256(str(source).encode()).hexdigest()[:10]}"
        meta = DocumentMeta(
            doc_id=doc_id,
            title=title or source.stem.replace("_", " ").title(),
            source_uri=str(source),
            doc_type=doc_type,
            owner=owner,
            acl_roles=acl_roles or list(DEFAULT_UPLOAD_ACL),
            created_date=date.today(),
            effective_date=effective_date or date.today(),
            version=version,
            authority_score=authority_score,
            policy_family=policy_family,
            supersedes=supersedes,
            checksum=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        )

        self.db.upsert_document(meta)
        if supersedes and self.db.get_document(supersedes):
            self.db.set_superseded_by(supersedes, doc_id)

        chunks = chunk_document(
            text,
            meta,
            target_chars=self.settings.chunk_target_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        self.db.insert_chunks(chunks)
        logger.info("Ingested %s as %s (%d chunks)", source.name, doc_id, len(chunks))
        return meta, chunks


def _as_date(value: Any) -> date:
    """Coerce a YAML scalar into a :class:`datetime.date`."""
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc
