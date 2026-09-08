"""
Baseline Comparison for AHRAG
================================
Runs a side-by-side comparison of AHRAG's governance-aware router against
the baseline routing strategies on the seeded evaluation set.

This script uses the REAL AHRAG evaluation harness — the same one used by
`python -m ahrag.evaluate` — but adds:
  1. Per-query detailed comparison table
  2. Route distribution analysis across systems
  3. Bootstrap confidence intervals on all metrics
  4. Paired bootstrap significance tests (AHRAG vs each baseline)
  5. Export of per-query results for further analysis

SYSTEMS COMPARED:
  B1: Fixed BM25 (always R1)
  B2: Fixed Dense (always R2)
  B3: Fixed Hybrid RRF (always R3)
  B4: Always-maximal iterative (always R4)
  B5: Complexity-only router (Adaptive-RAG ablation)
  P1: AHRAG governance-aware router (proposed system)

USAGE:
  python improvement_files/baseline_code/compare_baselines.py
  python improvement_files/baseline_code/compare_baselines.py --systems P1,B5,B3
  python improvement_files/baseline_code/compare_baselines.py --output comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_TODAY = date(2026, 8, 19)


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(1729)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    bootstrap_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    return (
        float(np.mean(values)),
        float(np.percentile(bootstrap_means, alpha * 100)),
        float(np.percentile(bootstrap_means, (1 - alpha) * 100)),
    )


def paired_bootstrap_test(a_scores, b_scores, n_bootstrap=10000):
    """Paired bootstrap test: p-value that A is NOT better than B."""
    rng = np.random.RandomState(1729)
    n = len(a_scores)
    count = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if np.mean(a_scores[idx]) <= np.mean(b_scores[idx]):
            count += 1
    return count / n_bootstrap


def run_comparison(system_keys: list[str] | None = None):
    """
    Run the full AHRAG evaluation with all systems and produce a comparison.

    Returns: (per_query_rows, system_summaries, per_system_rows)
    """
    from ahrag.config import RouterConfig, Settings
    from ahrag.db import Database
    from ahrag.eval.dataset import load_eval_set
    from ahrag.eval.metrics import aggregate
    from ahrag.eval.systems import build_systems
    from ahrag.evaluate import run_item
    from ahrag.pipeline import AHRAGEngine

    # Boot the shared engine
    print("[1/4] Initialising AHRAG engine...")
    settings = Settings()
    config = RouterConfig.load(settings.router_config)
    db = Database(settings.db_path)

    # Build a seeded engine for the database
    seed_engine = AHRAGEngine(settings=settings, config=config, db=db, today=EVAL_TODAY)
    seed_engine.ensure_seeded()
    info = seed_engine.backend_info()
    print(f"  Corpus: {info.get('documents', '?')} documents, {db.count_chunks()} chunks")
    print()

    # Build all 6 systems
    print("[2/4] Building evaluation systems...")
    all_systems = build_systems(settings, config, db, today=EVAL_TODAY)

    if system_keys:
        systems = [s for s in all_systems if s.key in system_keys]
        if not systems:
            print(f"ERROR: No matching systems for {system_keys}")
            print(f"Available: {[s.key for s in all_systems]}")
            sys.exit(1)
    else:
        systems = all_systems

    print(f"  Systems: {', '.join(s.key for s in systems)}")
    print()

    # Load evaluation set
    items = load_eval_set()
    print(f"  Evaluation items: {len(items)}")
    print()

    # Run evaluation
    print("[3/4] Running evaluation...")
    all_rows: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}

    for system in systems:
        started = time.perf_counter()
        rows = [run_item(system, item, system.engine) for item in items]
        elapsed = time.perf_counter() - started

        all_rows[system.key] = rows
        summaries[system.key] = aggregate(rows)

        # Quick summary line
        s = summaries[system.key]
        recall = s.get("recall_at_5")
        recall_str = f"{recall:.3f}" if recall is not None else "n/a"
        acl = s.get("acl_violation_rate", 0)
        abst = s.get("abstention_appropriateness", 0)
        print(f"  {system.key:4s} R@5={recall_str:>5s}  ACL-v={acl:.3f}  Abst-ok={abst:.3f}  ({elapsed:.2f}s)")

    print()
    return all_rows, summaries, systems


def print_detailed_comparison(all_rows, summaries, systems):
    """Print a detailed comparison report with CIs and significance tests."""

    print("=" * 94)
    print("DETAILED COMPARISON REPORT")
    print("=" * 94)
    print()

    # Metric comparison table with CIs
    metrics = [
        ("recall_at_5", "Recall@5"),
        ("recall_at_10", "Recall@10"),
        ("mrr", "MRR"),
        ("ndcg_at_10", "nDCG@10"),
        ("abstention_appropriateness", "Abstention-ok"),
        ("acl_violation_rate", "ACL Violation"),
    ]

    print(f"{'System':6s} {'Metric':20s} {'Mean':>7s} {'95% CI':>18s}")
    print("-" * 55)

    for system in systems:
        for metric_key, metric_name in metrics:
            # Get per-query values
            rows = all_rows[system.key]
            values = [r.get(metric_key, 0) for r in rows if r.get(metric_key) is not None]
            if not values:
                continue

            values_arr = np.array(values, dtype=np.float64)
            mean, lo, hi = bootstrap_ci(values_arr)
            print(f"{system.key:6s} {metric_name:20s} {mean:7.4f} [{lo:.4f}, {hi:.4f}]")
        print()

    # Route distribution comparison
    print("-" * 55)
    print("ROUTE DISTRIBUTION")
    print("-" * 55)

    route_names = ["R0", "R1", "R2", "R3", "R4", "ERROR"]
    print(f"{'':6s}", end="")
    for r in route_names:
        print(f" {r:>5s}", end="")
    print()

    for system in systems:
        counts = {r: 0 for r in route_names}
        for row in all_rows[system.key]:
            route = row.get("route", "ERROR")
            if route in counts:
                counts[route] += 1
            else:
                counts["ERROR"] += 1

        print(f"{system.key:6s}", end="")
        for r in route_names:
            print(f" {counts[r]:5d}", end="")
        print()

    # Significance tests: P1 vs each baseline
    p1_rows = all_rows.get("P1")
    if not p1_rows:
        print("\n  (P1 not in systems — skipping significance tests)")
        return

    print()
    print("-" * 55)
    print("SIGNIFICANCE TESTS (P1 vs each baseline)")
    print("-" * 55)
    print(f"{'Comparison':20s} {'Metric':15s} {'Delta':>8s} {'p-value':>8s} {'Sig?':>6s}")

    for system in systems:
        if system.key == "P1":
            continue

        for metric_key, metric_name in [("recall_at_5", "R@5"), ("mrr", "MRR")]:
            p1_values = np.array([
                r.get(metric_key, 0) for r in p1_rows
                if r.get(metric_key) is not None
            ], dtype=np.float64)
            base_values = np.array([
                r.get(metric_key, 0) for r in all_rows[system.key]
                if r.get(metric_key) is not None
            ], dtype=np.float64)

            if len(p1_values) == 0 or len(base_values) == 0:
                continue

            # Align lengths
            n = min(len(p1_values), len(base_values))
            p1_v = p1_values[:n]
            base_v = base_values[:n]

            diff = float(np.mean(p1_v) - np.mean(base_v))
            p_val = paired_bootstrap_test(p1_v, base_v)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "no"

            print(f"P1 vs {system.key:14s} {metric_name:15s} {diff:+8.4f} {p_val:8.4f} {sig:>6s}")

    # Per-query comparison (P1 vs B5 specifically — the key ablation)
    b5_rows = all_rows.get("B5")
    if b5_rows:
        print()
        print("-" * 94)
        print("PER-QUERY COMPARISON: P1 (Governance-Aware) vs B5 (Complexity-Only)")
        print("-" * 94)
        print(f"{'ID':12s} {'P1->':>4s} {'B5->':>4s} {'P1-R@5':>7s} {'B5-R@5':>7s} {'Delta':>7s} {'Notes'}")

        for p1_row, b5_row in zip(p1_rows, b5_rows):
            p1_route = p1_row.get("route", "?")
            b5_route = b5_row.get("route", "?")
            p1_r5 = p1_row.get("recall_at_5", 0) or 0
            b5_r5 = b5_row.get("recall_at_5", 0) or 0
            delta = p1_r5 - b5_r5
            query_id = p1_row.get("query_id", "?")

            notes = ""
            if p1_row.get("acl_violation"):
                notes += "ACL-VIOLATION "
            if b5_row.get("acl_violation"):
                notes += "B5-ACL-VIOLATION "
            if delta > 0:
                notes += "P1-WINS"
            elif delta < 0:
                notes += "B5-WINS"
            else:
                notes += "TIE"

            delta_str = f"{delta:+.3f}" if delta != 0 else "  0.000"
            p1_r5_str = f"{p1_r5:.3f}" if p1_r5 is not None else "  n/a"
            b5_r5_str = f"{b5_r5:.3f}" if b5_r5 is not None else "  n/a"

            print(f"{query_id:12s} {p1_route:>4s} {b5_route:>4s} {p1_r5_str:>7s} {b5_r5_str:>7s} {delta_str:>7s} {notes}")


def main():
    parser = argparse.ArgumentParser(description="Compare AHRAG against baselines")
    parser.add_argument(
        "--systems", type=str, default=None,
        help="Comma-separated system keys to compare (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save per-query results to JSON file",
    )
    args = parser.parse_args()

    system_keys = args.systems.split(",") if args.systems else None

    print("=" * 94)
    print("AHRAG Baseline Comparison")
    print("=" * 94)
    print()

    all_rows, summaries, systems = run_comparison(system_keys)

    print("[4/4] Generating comparison report...\n")
    print_detailed_comparison(all_rows, summaries, systems)

    if args.output:
        output_path = Path(args.output)
        output_data = {
            "reference_date": EVAL_TODAY.isoformat(),
            "systems": {s.key: {"name": s.name, "description": s.description} for s in systems},
            "summaries": summaries,
            "per_query": {key: rows for key, rows in all_rows.items()},
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    print(f"\nComparison complete.")


if __name__ == "__main__":
    main()
