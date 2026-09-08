"""Evaluation harness: labelled dataset, metrics, and comparable systems."""

from .dataset import EvalItem, load_eval_set
from .metrics import (
    aggregate,
    citation_scores,
    dcg,
    mrr,
    ndcg_at_k,
    percentile,
    recall_at_k,
)
from .systems import SystemSpec, build_systems

__all__ = [
    "EvalItem",
    "SystemSpec",
    "aggregate",
    "build_systems",
    "citation_scores",
    "dcg",
    "load_eval_set",
    "mrr",
    "ndcg_at_k",
    "percentile",
    "recall_at_k",
]
