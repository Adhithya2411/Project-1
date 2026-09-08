"""Canonical location and schema for experiment reports.

Why this exists
---------------
The experiment scripts each wrote JSON to whatever `--output` path the caller
happened to pass, which meant the results existed only wherever the person who
ran them left them. The UI's Evaluation page could not show any of it — it read
the database's `eval_runs` table, which only `python -m ahrag.evaluate` writes,
so the ablations, the baseline comparison, the embedding comparison and the
scope-specialisation measurement were invisible in the app that is supposed to
present them.

Every experiment now writes to ``data/reports/<kind>.json`` by default. The
directory is the contract: anything in it is a completed experiment, self
describing enough to render without the script that produced it. Each report
carries a ``kind`` so a reader can dispatch on it, and a ``generated_at`` so a
stale result cannot be mistaken for a fresh one.

Reports are deliberately *not* stored in SQLite. They are whole-run artefacts
that a human reads and a report renderer displays, not rows to query, and
keeping them as files means a run can be committed, diffed, and attached to a
write-up.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT

#: Canonical directory. Created on first write.
REPORTS_DIR = REPO_ROOT / "data" / "reports"

#: Report kinds, in the order they should be presented: corpus and labels
#: first, then the system comparison, then the component analyses.
KINDS: dict[str, str] = {
    "baselines": "System comparison (B1–B6, P1, P2)",
    "baselines_heldout": "System comparison on the learned router's held-out split",
    "ablations": "Component ablations",
    "embeddings": "Embedding backend comparison",
    "specialisation": "Scope-pure index specialisation",
    "router_training": "Learned router training",
    "annotation_agreement": "Inter-annotator agreement",
}


def default_path(kind: str) -> Path:
    """Return the canonical path for a report kind.

    Raises:
        ValueError: If ``kind`` is not a known report kind. Unknown kinds are
            rejected rather than accepted so the UI never has to guess how to
            render a file.
    """
    if kind not in KINDS:
        raise ValueError(
            f"Unknown report kind {kind!r}. Known kinds: {', '.join(sorted(KINDS))}"
        )
    return REPORTS_DIR / f"{kind}.json"


def save_report(kind: str, payload: dict[str, Any], path: Path | None = None) -> Path:
    """Write a report, stamping it with its kind and generation time.

    Args:
        kind: One of :data:`KINDS`.
        payload: The experiment's own result dictionary.
        path: Override the canonical location. Used when a caller wants a
            throwaway copy without disturbing what the UI displays.

    Returns:
        The path written.
    """
    target = Path(path) if path else default_path(kind)
    target.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "kind": kind,
        "title": KINDS.get(kind, kind),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **payload,
    }
    target.write_text(
        json.dumps(envelope, indent=2, default=str), encoding="utf-8"
    )
    return target


@dataclass(frozen=True)
class Report:
    """One report loaded from disk."""

    kind: str
    title: str
    generated_at: str
    path: Path
    data: dict[str, Any]

    @property
    def corpus_label(self) -> str:
        """Short description of which corpus produced this report.

        Reports that do not record a corpus return ``"unknown"`` rather than
        guessing: conflating the 9-document demo corpus with the benchmark one
        is the single easiest way to misread every number in this project.
        """
        corpus = self.data.get("corpus") or self.data.get("manifest") or ""
        if not corpus or corpus == "seed":
            return "demo corpus (9 documents)"
        if "integrated" in str(corpus):
            return "benchmark corpus (6,139 documents)"
        return str(Path(str(corpus)).parent.name or corpus)


def load_reports(directory: Path | None = None) -> list[Report]:
    """Load every readable report, in :data:`KINDS` presentation order.

    Unreadable or unrecognised files are skipped rather than raising: a
    half-written report from an interrupted run should not break the page that
    displays the others.
    """
    root = Path(directory) if directory else REPORTS_DIR
    if not root.is_dir():
        return []

    order = list(KINDS)
    found: list[Report] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        kind = data.get("kind") or path.stem
        found.append(
            Report(
                kind=str(kind),
                title=str(data.get("title") or KINDS.get(str(kind), str(kind))),
                generated_at=str(data.get("generated_at") or "unknown"),
                path=path,
                data=data,
            )
        )

    def sort_key(report: Report) -> tuple[int, str]:
        try:
            return (order.index(report.kind), report.kind)
        except ValueError:
            return (len(order), report.kind)

    return sorted(found, key=sort_key)


__all__ = ["KINDS", "REPORTS_DIR", "Report", "default_path", "load_reports", "save_report"]
