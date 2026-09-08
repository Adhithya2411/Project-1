"""AHRAG evaluation runner.

Usage::

    python -m ahrag.evaluate                    # run everything, print a report
    python -m ahrag.evaluate --json out.json    # also write machine-readable results
    python -m ahrag.evaluate --systems P1,B3    # run a subset
    python -m ahrag.evaluate --per-type         # add per-query-type breakdowns

Every number printed comes from an actual pipeline execution against the seeded
corpus. Nothing is estimated or carried over from a previous run. The one
exception is clearly labelled: ``est. cost/query`` is a token-count estimate
priced with local constants, because the default extractive generator makes no
API calls and therefore has no billed cost to report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .config import RouterConfig, Settings
from .db import Database
from .eval.dataset import EvalItem, load_eval_set
from .eval.metrics import (
    abstention_appropriate,
    acl_violation,
    aggregate,
    aggregate_by,
    citation_scores,
    freshness_compliant,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from .eval.systems import SystemSpec, build_systems
from .models import Route
from .pipeline import AHRAGEngine

logger = logging.getLogger(__name__)

# Fixed reference date so freshness-sensitive labels stay valid as time passes.
# The seeded corpus has documents effective up to 2026-07-10.
EVAL_TODAY = date(2026, 8, 19)


def run_item(system: SystemSpec, item: EvalItem, engine: AHRAGEngine) -> dict[str, Any]:
    """Execute one evaluation item against one system and score it.

    Returns:
        A per-query result row. A pipeline exception is captured into the row
        rather than aborting the run: a system that crashes on one query should
        show up as a scored failure, not disappear from the comparison.
    """
    user = engine.get_user(item.user_id)
    scope = engine.acl.scope_for(user)
    authorised = set(scope.allowed_chunk_ids)

    started = time.perf_counter()
    try:
        result = engine.answer(item.query, item.user_id, write_audit=False)
        error: str | None = None
    except Exception as exc:  # noqa: BLE001 - a crash is a result, not a stop
        logger.error("System %s failed on %s: %s", system.key, item.id, exc)
        elapsed = time.perf_counter() - started
        return {
            "system": system.key,
            "query_id": item.id,
            "query_type": item.query_type,
            "user_id": item.user_id,
            "route": "ERROR",
            "error": str(exc),
            "abstained": True,
            "abstention_appropriate": item.should_abstain,
            "acl_violation": False,
            "latency_s": round(elapsed, 4),
            "estimated_cost_usd": 0.0,
            "route_matches_expected": False,
        }

    evidence_ids = [e.chunk_id for e in result.evidence]
    citation_ids = [c.chunk_id for c in result.citations]
    gold = item.gold_set

    # Retrieval metrics are scored against the evidence pack the user actually
    # saw, not an internal candidate list, so they reflect what was served.
    violated, offenders = acl_violation(
        evidence_ids, citation_ids, set(item.forbidden_chunks), authorised
    )
    superseded = {
        e.chunk_id for e in result.evidence if e.is_superseded
    }
    citations = citation_scores(citation_ids, gold, evidence_ids)

    row: dict[str, Any] = {
        "system": system.key,
        "query_id": item.id,
        "query_type": item.query_type,
        "user_id": item.user_id,
        "route": result.decision.route.value,
        "expected_route": item.expected_route.value if item.expected_route else None,
        "route_matches_expected": (
            item.expected_route is not None
            and result.decision.route is item.expected_route
        ),
        "router_confidence": result.decision.confidence,
        "recall_at_5": recall_at_k(evidence_ids, gold, 5),
        "recall_at_10": recall_at_k(evidence_ids, gold, 10),
        "mrr": mrr(evidence_ids, gold),
        "ndcg_at_10": ndcg_at_k(evidence_ids, gold, 10),
        "abstained": result.abstained,
        "abstention_reason": result.abstention_reason.value,
        "abstention_appropriate": abstention_appropriate(
            item.should_abstain, result.abstained
        ),
        "acl_violation": violated,
        "acl_offenders": offenders,
        "acl_pool_size": len(authorised),
        "freshness_compliant": freshness_compliant(
            item.freshness_sensitive, citation_ids, superseded, result.abstained
        ),
        "conflicts_disclosed": len(result.conflicts),
        "evidence_count": len(evidence_ids),
        "citation_count": len(citation_ids),
        "latency_s": round(result.total_latency_s, 4),
        "estimated_cost_usd": result.estimated_cost_usd,
        "error": error,
    }
    row.update(citations)
    return row


def run_system(
    system: SystemSpec, items: Sequence[EvalItem]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run every item against one system and aggregate the results."""
    rows = [run_item(system, item, system.engine) for item in items]
    return rows, aggregate(rows)


def _format_optional(value: Any, spec: str = ".3f") -> str:
    """Format a possibly-None metric for the report table."""
    if value is None:
        return "  n/a"
    return format(value, spec)


def print_report(
    summaries: dict[str, dict[str, Any]],
    systems: Sequence[SystemSpec],
    per_type: dict[str, dict[str, dict[str, Any]]] | None,
    item_count: int,
    backend_info: dict[str, Any],
) -> None:
    """Print the human-readable evaluation report."""
    print()
    print("=" * 94)
    print("AHRAG EVALUATION REPORT")
    print("=" * 94)
    print(f"Items: {item_count}   Reference date: {EVAL_TODAY.isoformat()}")
    print(
        "Backends: "
        + ", ".join(f"{k}={v}" for k, v in backend_info.items() if k != "chunks")
    )
    print()
    print("Systems:")
    for system in systems:
        print(f"  {system.key}  {system.name}")
        print(f"      {system.description}")
    print()

    header = (
        f"{'':4} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG':>6} "
        f"{'CitP':>6} {'CitCov':>7} {'Grnd':>6} {'Abst✓':>6} "
        f"{'ACLv':>5} {'Fresh':>6} {'MeanS':>7} {'p95S':>7} {'$/q':>9}"
    )
    print("-" * 94)
    print("RETRIEVAL / CITATION / GOVERNANCE / EFFICIENCY")
    print("-" * 94)
    print(header)
    for system in systems:
        s = summaries[system.key]
        print(
            f"{system.key:4} "
            f"{_format_optional(s['recall_at_5']):>6} "
            f"{_format_optional(s['recall_at_10']):>6} "
            f"{_format_optional(s['mrr']):>6} "
            f"{_format_optional(s['ndcg_at_10']):>6} "
            f"{_format_optional(s['citation_precision']):>6} "
            f"{_format_optional(s['citation_coverage']):>7} "
            f"{_format_optional(s['citation_groundedness']):>6} "
            f"{s['abstention_appropriateness']:>6.3f} "
            f"{s['acl_violations']:>5d} "
            f"{_format_optional(s['freshness_compliance']):>6} "
            f"{s['mean_latency_s']:>7.4f} "
            f"{s['p95_latency_s']:>7.4f} "
            f"{s['mean_cost_usd']:>9.6f}"
        )
    print()
    print("  R@5/R@10/MRR/nDCG are over answerable items only "
          f"(n={summaries[systems[0].key]['recall_at_5_n']}).")
    print("  CitP/CitCov are over items that produced citations; Grnd = share of")
    print("  citations resolvable in the served evidence pack (must be 1.000).")
    print("  ACLv = count of items where unauthorised material surfaced (must be 0).")
    print("  $/q is an ESTIMATE from token counts; the default generator is offline")
    print("  and incurs no actual API cost.")
    print()

    print("-" * 94)
    print("ROUTE DISTRIBUTION")
    print("-" * 94)
    print(f"{'':4} {'R0':>6} {'R1':>6} {'R2':>6} {'R3':>6} {'R4':>6}   {'AbstRate':>9}")
    for system in systems:
        s = summaries[system.key]
        dist = s["route_distribution"]
        print(
            f"{system.key:4} "
            + " ".join(f"{dist[r]:>6d}" for r in ("R0", "R1", "R2", "R3", "R4"))
            + f"   {s['abstention_rate']:>9.3f}"
        )
    print()

    if per_type:
        print("-" * 94)
        print("PER QUERY TYPE — Recall@5 / abstention appropriateness / mean latency (s)")
        print("-" * 94)
        type_names = sorted({t for sysd in per_type.values() for t in sysd})
        print(f"{'':4} " + " ".join(f"{t[:14]:>16}" for t in type_names))
        for system in systems:
            cells = []
            for type_name in type_names:
                stats = per_type[system.key].get(type_name)
                if not stats:
                    cells.append(f"{'—':>16}")
                    continue
                recall = _format_optional(stats["recall_at_5"], ".2f")
                cells.append(
                    f"{recall}/{stats['abstention_appropriateness']:.2f}/"
                    f"{stats['mean_latency_s']:.3f}".rjust(16)
                )
            print(f"{system.key:4} " + " ".join(cells))
        print()

    print("=" * 94)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m ahrag.evaluate``."""
    parser = argparse.ArgumentParser(
        prog="python -m ahrag.evaluate",
        description="Run the AHRAG evaluation suite across all comparable systems.",
    )
    parser.add_argument("--json", type=Path, help="Write full results to this JSON file.")
    parser.add_argument(
        "--systems",
        type=str,
        default="",
        help="Comma-separated system keys to run (default: all). e.g. P1,B3",
    )
    parser.add_argument(
        "--per-type", action="store_true", help="Include per-query-type breakdowns."
    )
    parser.add_argument(
        "--eval-set", type=Path, default=None, help="Path to an alternative eval set."
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Path to an alternative corpus manifest.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite path (default: a dedicated evaluation database).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = Settings()
    config = RouterConfig.load(settings.router_config)

    # A dedicated evaluation database keeps benchmark runs from mixing with the
    # corpus and audit trail a reviewer is inspecting in the UI.
    db_path = args.db or (Path(settings.data_dir) / "ahrag_eval.sqlite3")
    db = Database(db_path)

    try:
        items = load_eval_set(args.eval_set)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: could not load evaluation set: {exc}", file=sys.stderr)
        return 2

    # Seed once, then share the corpus and indexes across all systems.
    bootstrap = AHRAGEngine(settings=settings, config=config, db=db, today=EVAL_TODAY)
    seed_result = bootstrap.seed(reset=True, manifest_path=args.manifest)
    if seed_result.errors:
        print("warning: corpus seeding reported errors:", file=sys.stderr)
        for message in seed_result.errors:
            print(f"  - {message}", file=sys.stderr)

    systems = build_systems(settings, config, db, today=EVAL_TODAY)
    if args.systems:
        wanted = {key.strip().upper() for key in args.systems.split(",") if key.strip()}
        unknown = wanted - {s.key for s in systems}
        if unknown:
            print(
                f"error: unknown system key(s): {', '.join(sorted(unknown))}",
                file=sys.stderr,
            )
            return 2
        systems = [s for s in systems if s.key in wanted]

    run_id = uuid.uuid4().hex[:12]
    config_hash = hashlib.sha256(
        json.dumps(config.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()[:16]
    db.start_eval_run(run_id, config_hash, notes=f"{len(items)} items")

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    per_type: dict[str, dict[str, dict[str, Any]]] = {}

    for system in systems:
        print(f"running {system.key} ({system.name})...", file=sys.stderr)
        rows, summary = run_system(system, items)
        summaries[system.key] = summary
        per_type[system.key] = aggregate_by(rows, "query_type")
        all_rows.extend(rows)
        for row in rows:
            db.write_eval_result(run_id, system.key, row["query_id"], row)

    db.finish_eval_run(run_id, {"summaries": summaries, "config_hash": config_hash})

    print_report(
        summaries,
        systems,
        per_type if args.per_type else None,
        len(items),
        bootstrap.backend_info(),
    )

    # Governance invariant. A non-zero count here is a hard failure of the
    # central claim, so it changes the exit code rather than only printing.
    total_violations = sum(s["acl_violations"] for s in summaries.values())
    if total_violations:
        print(
            f"FAIL: {total_violations} ACL violation(s) detected across systems.",
            file=sys.stderr,
        )
        for row in all_rows:
            if row.get("acl_violation"):
                print(
                    f"  {row['system']}/{row['query_id']}: {row.get('acl_offenders')}",
                    file=sys.stderr,
                )
    else:
        print("ACL violation rate is zero across all systems and all items. ✓")

    if args.json:
        payload = {
            "run_id": run_id,
            "config_hash": config_hash,
            "reference_date": EVAL_TODAY.isoformat(),
            "backends": bootstrap.backend_info(),
            "item_count": len(items),
            "systems": [
                {"key": s.key, "name": s.name, "description": s.description}
                for s in systems
            ],
            "summaries": summaries,
            "per_query_type": per_type,
            "rows": all_rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Wrote full results to {args.json}")

    return 1 if total_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
