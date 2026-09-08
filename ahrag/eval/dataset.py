"""Labelled evaluation dataset loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..models import Route

SEED_DIR = Path(__file__).resolve().parent.parent / "seed"


class EvalItem(BaseModel):
    """One labelled evaluation query."""

    id: str
    query: str
    user_id: str
    query_type: str
    expected_route: Route | None = None
    gold_chunks: list[str] = Field(default_factory=list)
    forbidden_chunks: list[str] = Field(
        default_factory=list,
        description=(
            "Chunks that would answer the question but are outside this user's "
            "scope. Any appearance in evidence or citations is an ACL violation."
        ),
    )
    should_abstain: bool = False
    freshness_sensitive: bool = False
    notes: str = ""

    @property
    def gold_set(self) -> set[str]:
        """Gold chunk IDs as a set."""
        return set(self.gold_chunks)


def load_eval_set(path: str | Path | None = None) -> list[EvalItem]:
    """Load and validate the labelled evaluation suite.

    Args:
        path: YAML file. Defaults to the packaged ``seed/eval_set.yaml``.

    Returns:
        The parsed items, in file order.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the file is malformed, empty, or contains duplicate IDs.
    """
    target = Path(path) if path else SEED_DIR / "eval_set.yaml"
    if not target.exists():
        raise FileNotFoundError(f"Evaluation set not found: {target}")
    try:
        raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Evaluation set {target} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict) or "items" not in raw:
        raise ValueError(f"Evaluation set {target} must contain an 'items' list.")

    items = [EvalItem.model_validate(entry) for entry in raw["items"]]
    if not items:
        raise ValueError(f"Evaluation set {target} contains no items.")

    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"Duplicate evaluation item id: {item.id}")
        seen.add(item.id)
        # A consistency check on the labels themselves: an item that should be
        # answered needs gold evidence, and an item that should be refused must
        # not have any. A contradictory label would silently corrupt every
        # metric derived from it.
        if item.should_abstain and item.gold_chunks:
            raise ValueError(
                f"Item {item.id} is labelled should_abstain but has gold_chunks."
            )
        if not item.should_abstain and not item.gold_chunks:
            raise ValueError(
                f"Item {item.id} is answerable but has no gold_chunks."
            )
    return items


def query_types(items: list[EvalItem]) -> list[str]:
    """Return the distinct query types present, in stable order."""
    seen: list[str] = []
    for item in items:
        if item.query_type not in seen:
            seen.append(item.query_type)
    return seen
