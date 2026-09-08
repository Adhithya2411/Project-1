"""Shared corpus/eval-set plumbing for the experiment scripts.

Why this exists
---------------
Every script in ``improvement_files/`` needs the same four things: pick a
corpus, pick an evaluation set, build a seeded engine, and check that the labels
actually refer to chunks that exist. Before this module each script did its own
version, and three of them did it wrong in the same way — calling
``load_eval_set()`` with no argument, which silently defaults to the 32-item
packaged seed set no matter what corpus was requested. Every reported number
came from the demo corpus while the scripts printed messages about the
integrated one.

Two functions are the load-bearing ones:

* :func:`validate_eval_set` — ``load_eval_set`` checks label *consistency* (an
  abstention item must not carry gold chunks) but not label *existence*. A gold
  chunk ID that does not exist reads out as recall 0.0, so a broken converter
  looks like a badly-performing system. This turns that into an error.

* :func:`build_seeded_engine` — ingesting and indexing a 6,000-document corpus
  takes minutes, and it happens identically for every script and every
  configuration sweep. The corpus is content-addressed by manifest digest and
  cached on disk, so the cost is paid once per corpus rather than once per run.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from ..config import REPO_ROOT, RouterConfig, Settings
from ..db import Database
from .dataset import EvalItem, load_eval_set

logger = logging.getLogger(__name__)

# Fixed reference date, so freshness-dependent behaviour does not drift as the
# real clock passes a document's effective date. Matches ahrag.evaluate.
EVAL_TODAY = date(2026, 8, 19)

INTEGRATED_DIR = REPO_ROOT / "improvement_files" / "datasets" / "integrated"
INTEGRATED_MANIFEST = INTEGRATED_DIR / "manifest.yaml"
INTEGRATED_EVAL_SET = INTEGRATED_DIR / "eval_set.yaml"

CACHE_ROOT = REPO_ROOT / "data" / "corpus-cache"


class DanglingLabelError(ValueError):
    """Raised when an evaluation label references a chunk that does not exist."""


# ---------------------------------------------------------------------------
# Argument plumbing
# ---------------------------------------------------------------------------


def add_corpus_arguments(parser: Any) -> None:
    """Add ``--manifest``, ``--eval-set`` and ``--integrated`` to a parser.

    Every experiment script takes the same three, so they are defined once.
    """
    group = parser.add_argument_group("corpus selection")
    group.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Corpus manifest YAML. Default: the packaged 9-document seed corpus.",
    )
    group.add_argument(
        "--eval-set",
        type=str,
        default=None,
        help="Evaluation set YAML. Default: the packaged 32-item seed suite.",
    )
    group.add_argument(
        "--integrated",
        action="store_true",
        help=(
            "Shorthand for the integrated benchmark corpus and evaluation set "
            "produced by improvement_files/datasets/integrate_datasets.py."
        ),
    )
    group.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Re-ingest the corpus even if a cached database exists.",
    )


def resolve_corpus(args: Any) -> tuple[Path | None, Path | None]:
    """Resolve corpus arguments into ``(manifest, eval_set)`` paths.

    ``None`` means "use the packaged seed default", which is what
    ``IngestionService.seed_from_manifest`` and ``load_eval_set`` already do.

    Raises:
        FileNotFoundError: If an explicitly requested file is missing, or if
            ``--integrated`` is given before the integration script has run.
    """
    if getattr(args, "integrated", False):
        if not INTEGRATED_MANIFEST.exists() or not INTEGRATED_EVAL_SET.exists():
            raise FileNotFoundError(
                "The integrated corpus is not built. Run:\n"
                "  python improvement_files/datasets/integrate_datasets.py"
            )
        return INTEGRATED_MANIFEST, INTEGRATED_EVAL_SET

    manifest = Path(args.manifest).resolve() if getattr(args, "manifest", None) else None
    eval_set = Path(args.eval_set).resolve() if getattr(args, "eval_set", None) else None
    for label, path in (("manifest", manifest), ("eval set", eval_set)):
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    return manifest, eval_set


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------


def validate_eval_set(
    items: Sequence[EvalItem], db: Database, strict: bool = True
) -> dict[str, Any]:
    """Check that every gold and forbidden chunk ID exists in ``db``.

    Args:
        items: The loaded evaluation items.
        db: A seeded database.
        strict: Raise on any dangling reference. Set False to get the report
            without failing, which is useful while repairing a converter.

    Returns:
        A summary with the dangling references found and per-stratum counts.

    Raises:
        DanglingLabelError: If ``strict`` and any reference does not resolve.
    """
    known = {chunk.chunk_id for chunk in db.get_chunks()}
    dangling: list[str] = []
    for item in items:
        for chunk_id in list(item.gold_chunks) + list(item.forbidden_chunks):
            if chunk_id not in known:
                dangling.append(f"{item.id} -> {chunk_id}")

    report = {
        "items": len(items),
        "corpus_chunks": len(known),
        "answerable": sum(1 for i in items if i.gold_chunks),
        "should_abstain": sum(1 for i in items if i.should_abstain),
        "freshness_sensitive": sum(1 for i in items if i.freshness_sensitive),
        "with_forbidden_chunks": sum(1 for i in items if i.forbidden_chunks),
        "dangling_references": len(dangling),
        "examples": dangling[:10],
    }

    if dangling and strict:
        raise DanglingLabelError(
            f"{len(dangling)} evaluation labels reference chunks that do not "
            f"exist in this corpus. The first few:\n  "
            + "\n  ".join(dangling[:10])
            + "\n\nThis usually means the evaluation set was built for a "
            "different corpus, or a converter guessed chunk IDs instead of "
            "resolving them through ahrag.ingestion.chunker."
        )
    return report


def describe_eval_set(items: Sequence[EvalItem]) -> dict[str, int]:
    """Count items per query type, for stratified reporting (§2d)."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item.query_type] = counts.get(item.query_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Engine construction with a content-addressed corpus cache
# ---------------------------------------------------------------------------


def _manifest_digest(manifest: Path | None, settings_key: str) -> str:
    """Content hash of a manifest plus the settings that affect chunking."""
    digest = hashlib.blake2b(digest_size=10)
    digest.update(settings_key.encode("utf-8"))
    if manifest is None:
        digest.update(b"seed-default")
        seed_dir = REPO_ROOT / "ahrag" / "seed"
        for path in sorted(seed_dir.rglob("*")):
            if path.is_file():
                digest.update(path.read_bytes())
    else:
        digest.update(manifest.read_bytes())
        corpus_dir = manifest.parent / "corpus"
        if corpus_dir.is_dir():
            # Hash file names and sizes rather than contents: a 4 MB corpus
            # would otherwise be re-read on every invocation just to decide
            # whether it changed.
            for path in sorted(corpus_dir.iterdir()):
                if path.is_file():
                    digest.update(path.name.encode("utf-8"))
                    digest.update(str(path.stat().st_size).encode("utf-8"))
    return digest.hexdigest()


def build_seeded_engine(
    manifest: Path | None = None,
    *,
    settings_overrides: dict[str, Any] | None = None,
    router: Any = None,
    today: date | None = EVAL_TODAY,
    rebuild_cache: bool = False,
    cache: bool = True,
    quiet: bool = False,
):
    """Return a seeded :class:`~ahrag.pipeline.AHRAGEngine`.

    The corpus is ingested once per ``(manifest, chunking settings)`` pair and
    the resulting SQLite file is reused. Index structures are rebuilt in-process
    on load — they are derived data and cheap relative to ingestion.

    Args:
        manifest: Corpus manifest, or None for the packaged seed corpus.
        settings_overrides: Extra ``Settings`` fields (embedding backend, SPIS
            flags, and so on). Fields that change chunking participate in the
            cache key; others do not, so an embedding sweep reuses one corpus.
        router: Optional router instance to inject.
        today: Fixed reference date for freshness comparisons.
        rebuild_cache: Ingest again even if a cached database exists.
        cache: Set False to always use a throwaway database.
        quiet: Suppress progress output.

    Returns:
        A seeded engine.
    """
    from ..pipeline import AHRAGEngine

    overrides = dict(settings_overrides or {})
    # Only chunking parameters change the stored corpus; everything else is
    # applied on top of a cached database without re-ingesting.
    chunk_key = "|".join(
        str(overrides.get(field, ""))
        for field in ("chunk_target_chars", "chunk_overlap_chars")
    )
    digest = _manifest_digest(manifest, chunk_key)

    if cache:
        cache_dir = CACHE_ROOT / digest
        db_path = cache_dir / "corpus.sqlite3"
        marker = cache_dir / "READY"
    else:
        import tempfile

        cache_dir = Path(tempfile.mkdtemp(prefix="ahrag-corpus-"))
        db_path = cache_dir / "corpus.sqlite3"
        marker = cache_dir / "READY"

    if rebuild_cache and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        data_dir=cache_dir,
        db_path=db_path,
        router_config=REPO_ROOT / "config" / "router.yaml",
        **overrides,
    )
    config = RouterConfig.load(settings.router_config)
    database = Database(settings.db_path)

    already_seeded = marker.exists() and database.count_chunks() > 0
    engine = AHRAGEngine(
        settings=settings, config=config, db=database, router=router, today=today
    )

    if not already_seeded:
        if not quiet:
            label = manifest.name if manifest else "packaged seed corpus"
            print(f"  ingesting {label} (first run for this corpus; cached afterwards)...")
        result = engine.seed(reset=True, manifest_path=manifest)
        if result.errors and not quiet:
            print(f"  WARNING: {len(result.errors)} ingestion errors; first: {result.errors[0]}")
        marker.write_text(
            f"documents={result.documents}\nchunks={result.chunks}\n", encoding="utf-8"
        )
        if not quiet:
            print(f"  ingested {result.documents} documents, {result.chunks} chunks")
    elif not quiet:
        print(f"  reusing cached corpus ({database.count_chunks()} chunks)")

    return engine


def load_and_validate(
    engine, eval_set: Path | None, strict: bool = True, quiet: bool = False
) -> tuple[list[EvalItem], dict[str, Any]]:
    """Load an evaluation set and validate it against ``engine``'s corpus.

    This is the pairing that the previous scripts got wrong: the evaluation set
    must be loaded from the path the caller asked for, and it must be checked
    against the corpus that is actually loaded.
    """
    items = load_eval_set(eval_set)
    report = validate_eval_set(items, engine.db, strict=strict)
    if not quiet:
        print(
            f"  eval set: {report['items']} items "
            f"({report['answerable']} answerable, "
            f"{report['should_abstain']} abstention, "
            f"{report['freshness_sensitive']} freshness, "
            f"{report['with_forbidden_chunks']} ACL-probe), "
            f"dangling refs: {report['dangling_references']}"
        )
    return items, report


__all__ = [
    "CACHE_ROOT",
    "DanglingLabelError",
    "EVAL_TODAY",
    "INTEGRATED_EVAL_SET",
    "INTEGRATED_MANIFEST",
    "add_corpus_arguments",
    "build_seeded_engine",
    "describe_eval_set",
    "load_and_validate",
    "resolve_corpus",
    "validate_eval_set",
]
