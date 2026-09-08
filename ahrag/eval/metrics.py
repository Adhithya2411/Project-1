"""Retrieval, citation, governance, and efficiency metrics.

Every metric here is computed from an actual pipeline execution. Nothing is
estimated, extrapolated, or filled in from a prior — an unmeasurable metric for
a given item is excluded from that item's aggregate rather than defaulted, and
the denominator is reported alongside so a reader can see how many items
actually contributed.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float | None:
    """Fraction of gold chunks appearing in the top ``k`` retrieved.

    Returns None when there are no gold chunks, so abstention items do not
    contribute a spurious 0.0 (or 1.0) to a retrieval average.
    """
    if not gold:
        return None
    top = set(retrieved[:k])
    return len(top & gold) / len(gold)


def mrr(retrieved: Sequence[str], gold: set[str]) -> float | None:
    """Reciprocal rank of the first gold chunk, or 0.0 if none is retrieved."""
    if not gold:
        return None
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain with log2 position discounting."""
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def ndcg_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float | None:
    """nDCG@k with binary relevance.

    Binary because the labels are binary: a chunk either contains the answer or
    it does not. Graded relevance would need a graded annotation the suite does
    not have, and inventing one would make the number look more precise than it
    is.
    """
    if not gold:
        return None
    gains = [1.0 if chunk_id in gold else 0.0 for chunk_id in retrieved[:k]]
    ideal = [1.0] * min(len(gold), k)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return None
    return dcg(gains) / ideal_dcg


# ---------------------------------------------------------------------------
# Citation metrics
# ---------------------------------------------------------------------------


def citation_scores(
    citations: Sequence[str], gold: set[str], evidence: Sequence[str]
) -> dict[str, float | None]:
    """Compute citation precision, coverage, and groundedness.

    ``precision``
        Share of cited chunks that are gold. Measures whether the answer cites
        the right passages.
    ``coverage``
        Share of gold chunks that were cited. Measures whether the answer used
        the evidence it should have.
    ``groundedness``
        Share of cited chunks that appear in the served evidence pack. This must
        be 1.0: a citation to something outside the pack is a fabricated or
        leaked reference, and the citation verifier is supposed to make it
        impossible. It is measured rather than assumed.
    """
    if not citations:
        return {
            "citation_precision": None,
            "citation_coverage": 0.0 if gold else None,
            "citation_groundedness": None,
        }
    cited = list(dict.fromkeys(citations))
    evidence_set = set(evidence)
    grounded = sum(1 for c in cited if c in evidence_set) / len(cited)
    if not gold:
        return {
            "citation_precision": None,
            "citation_coverage": None,
            "citation_groundedness": grounded,
        }
    precision = sum(1 for c in cited if c in gold) / len(cited)
    coverage = len(set(cited) & gold) / len(gold)
    return {
        "citation_precision": precision,
        "citation_coverage": coverage,
        "citation_groundedness": grounded,
    }


# ---------------------------------------------------------------------------
# Governance metrics
# ---------------------------------------------------------------------------


def acl_violation(
    evidence: Sequence[str],
    citations: Sequence[str],
    forbidden: set[str],
    authorised: set[str],
) -> tuple[bool, list[str]]:
    """Detect any use of unauthorised material.

    Two independent tests, because they catch different failures:

    1. Anything in ``forbidden`` (a labelled out-of-scope chunk) that appears in
       evidence or citations.
    2. Anything in evidence or citations that is not in ``authorised`` (the
       user's actual scope), which catches leaks the labels did not anticipate.

    Returns:
        ``(violated, offending_chunk_ids)``.
    """
    surfaced = set(evidence) | set(citations)
    offenders = sorted((surfaced & forbidden) | (surfaced - authorised))
    return bool(offenders), offenders


def abstention_appropriate(should_abstain: bool, did_abstain: bool) -> bool:
    """True when abstention behaviour matched the label.

    Symmetric on purpose: refusing an answerable question is as wrong as
    answering an unanswerable one. A system that abstains constantly would score
    perfectly on a one-sided metric while being useless.
    """
    return should_abstain == did_abstain


def freshness_compliant(
    freshness_sensitive: bool,
    citations: Sequence[str],
    superseded_chunks: set[str],
    abstained: bool,
) -> bool | None:
    """Check that a freshness-sensitive answer did not rest on a retired source.

    Compliance means the *top* cited source is current. Citing a superseded
    version alongside the current one is permitted and expected — that is the
    conflict-disclosure behaviour — so only the leading citation is tested.

    Returns None when the item is not freshness-sensitive, or when the system
    abstained (there is no answer whose currency could be judged).
    """
    if not freshness_sensitive or abstained:
        return None
    if not citations:
        return None
    return citations[0] not in superseded_chunks


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], pct: float) -> float:
    """Return the ``pct`` percentile using linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean(values: Sequence[float | None]) -> float | None:
    """Mean of the non-None values, or None when there are none."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query result rows into a summary.

    Args:
        rows: Per-query dicts produced by the runner.

    Returns:
        A summary dict. Averaged metrics carry an ``*_n`` companion giving the
        number of items that contributed, so a metric averaged over 3 items is
        not mistaken for one averaged over 30.
    """
    if not rows:
        return {"n": 0}

    summary: dict[str, Any] = {"n": len(rows)}

    for key in (
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_10",
        "citation_precision",
        "citation_coverage",
        "citation_groundedness",
    ):
        values = [row.get(key) for row in rows]
        present = [v for v in values if v is not None]
        summary[key] = round(mean(values), 4) if present else None
        summary[f"{key}_n"] = len(present)

    summary["abstention_appropriateness"] = round(
        sum(1 for row in rows if row.get("abstention_appropriate")) / len(rows), 4
    )
    summary["abstention_rate"] = round(
        sum(1 for row in rows if row.get("abstained")) / len(rows), 4
    )

    violations = sum(1 for row in rows if row.get("acl_violation"))
    summary["acl_violation_rate"] = round(violations / len(rows), 4)
    summary["acl_violations"] = violations

    fresh_values = [
        row.get("freshness_compliant")
        for row in rows
        if row.get("freshness_compliant") is not None
    ]
    summary["freshness_compliance"] = (
        round(sum(1 for v in fresh_values if v) / len(fresh_values), 4)
        if fresh_values
        else None
    )
    summary["freshness_compliance_n"] = len(fresh_values)

    latencies = [float(row.get("latency_s", 0.0)) for row in rows]
    summary["mean_latency_s"] = round(sum(latencies) / len(latencies), 4)
    summary["p95_latency_s"] = round(percentile(latencies, 95), 4)

    costs = [float(row.get("estimated_cost_usd", 0.0)) for row in rows]
    summary["mean_cost_usd"] = round(sum(costs) / len(costs), 8)
    summary["total_cost_usd"] = round(sum(costs), 8)

    route_counts = Counter(str(row.get("route", "?")) for row in rows)
    summary["route_distribution"] = {
        route: route_counts.get(route, 0) for route in ("R0", "R1", "R2", "R3", "R4")
    }

    summary["route_agreement"] = round(
        sum(1 for row in rows if row.get("route_matches_expected")) / len(rows), 4
    )
    return summary


def aggregate_by(
    rows: Sequence[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    """Aggregate rows grouped by ``row[key]``, in first-seen order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "?")), []).append(row)
    return {name: aggregate(group) for name, group in groups.items()}
