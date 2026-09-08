"""Head-to-head system comparison with the statistics improvement.txt §5 asks for.

Runs every system in ``ahrag.eval.systems`` over one shared corpus and reports,
for each pairwise comparison against the proposed system: the mean, a 95%
bootstrap confidence interval, the paired difference, a two-sided paired
bootstrap p-value, and Cohen's *d*. Also reports per-stratum results, because a
single pooled average over a suite containing ACL probes, unanswerable queries
and multi-hop questions hides more than it shows (§2d).

TWO CORRECTIONS TO THE PREVIOUS VERSION
---------------------------------------
1. **The paired test was not paired.** It filtered ``None`` values out of each
   system independently and then truncated both lists to the shorter length::

       p1_values = [r[m] for r in p1_rows if r[m] is not None]
       base_values = [r[m] for r in base_rows if r[m] is not None]
       n = min(len(p1_values), len(base_values))
       compare(p1_values[:n], base_values[:n])

   If the two systems have ``None`` at different positions — which happens as
   soon as one abstains where the other does not — this pairs query *i* of one
   system against query *j* of the other. The comparison silently stops being
   paired, which is the entire basis of the test. Here a single mask is
   computed across all systems and applied identically, and
   ``ahrag.stats.paired_bootstrap`` refuses misaligned input outright.

2. **The p-value was the wrong quantity.** It counted resamples where the
   difference was ``<= 0``, which estimates ``P(delta <= 0 | data)`` rather
   than ``P(data this extreme | H0)``. See the note in ``ahrag/stats.py``.

USAGE
  python improvement_files/baseline_code/compare_baselines.py --integrated
  python improvement_files/baseline_code/compare_baselines.py --integrated --systems P1,P2,B6
  python improvement_files/baseline_code/compare_baselines.py --integrated --by-type
  python improvement_files/baseline_code/compare_baselines.py --integrated --output baselines.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ahrag.eval.harness import (  # noqa: E402
    add_corpus_arguments,
    build_seeded_engine,
    load_and_validate,
    resolve_corpus,
)
from ahrag.stats import (  # noqa: E402
    bootstrap_ci,
    cohens_d,
    interpret_d,
    paired_bootstrap,
    significance_marker,
)

# Per-row keys from ``ahrag.evaluate.run_item``. ``higher_is_better`` matters
# for reading the sign of a delta; ACL violations are the one metric where a
# positive difference is bad.
METRICS: list[tuple[str, str, bool]] = [
    ("recall_at_5", "Recall@5", True),
    ("recall_at_10", "Recall@10", True),
    ("mrr", "MRR", True),
    ("ndcg_at_10", "nDCG@10", True),
    ("abstention_appropriate", "Abstention-ok", True),
    ("citation_coverage", "Citation coverage", True),
    ("freshness_compliant", "Freshness-ok", True),
    ("acl_violation", "ACL violations", False),
]

PROPOSED = "P1"


def collect_rows(engine, systems, items, verbose: bool = True):
    """Run every system over every item, sharing one index."""
    from ahrag.evaluate import run_item

    results: dict[str, list[dict]] = {}
    for system in systems:
        # All systems share the base engine's corpus, index and ACL snapshot;
        # only the router differs. Rebuilding the embedding space per system
        # would multiply the run time and change nothing.
        system.engine.index = engine.index
        system.engine.acl = engine.acl
        system.engine.retrieval.index = engine.index
        system.engine.retrieval.acl = engine.acl
        system.engine.validator.acl = engine.acl

        started = time.perf_counter()
        rows = [run_item(system, item, system.engine) for item in items]
        elapsed = time.perf_counter() - started
        results[system.key] = rows

        recall = [r["recall_at_5"] for r in rows if r["recall_at_5"] is not None]
        if verbose:
            mean_recall = f"{np.mean(recall):.4f}" if recall else "n/a"
            violations = sum(1 for r in rows if r["acl_violation"])
            errors = sum(1 for r in rows if r.get("error"))
            print(f"  {system.key:4s} {system.engine.router.name:34s} "
                  f"R@5={mean_recall:>6s}  ACL-v={violations:<3d} "
                  f"err={errors:<3d} ({elapsed:.1f}s)")
    return results


def aligned_vectors(
    results: dict[str, list[dict]], metric: str
) -> dict[str, np.ndarray]:
    """Extract item-aligned metric vectors across all systems.

    A single mask is built from the items where *every* system produced a
    value, then applied identically. This is the fix for the pairing bug
    described in the module docstring: filtering per system and truncating
    produces vectors of equal length that are no longer item-aligned.
    """
    keys = list(results)
    if not keys:
        return {}
    length = len(results[keys[0]])
    mask = np.ones(length, dtype=bool)
    for key in keys:
        rows = results[key]
        for index in range(length):
            value = rows[index].get(metric)
            if value is None:
                mask[index] = False
    return {
        key: np.array(
            [float(results[key][i][metric]) for i in range(length) if mask[i]],
            dtype=np.float64,
        )
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AHRAG against baselines")
    add_corpus_arguments(parser)
    parser.add_argument("--systems", type=str, default=None,
                        help="Comma-separated system keys (default: all)")
    parser.add_argument("--proposed", type=str, default=PROPOSED,
                        help="System used as the comparison point")
    parser.add_argument("--by-type", action="store_true",
                        help="Also report Recall@5 per query type (§2d)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    manifest, eval_set = resolve_corpus(args)

    print("=" * 100)
    print("AHRAG BASELINE COMPARISON")
    print("=" * 100)
    print()

    print("[1/4] Loading corpus...")
    engine = build_seeded_engine(manifest, rebuild_cache=args.rebuild_cache)
    items, label_report = load_and_validate(engine, eval_set)
    if args.limit:
        items = items[: args.limit]
        print(f"  limited to {len(items)} queries")
    print(f"  corpus: {len(engine.db.get_documents())} documents, "
          f"{engine.db.count_chunks()} chunks")
    print(f"  embedder: {engine.index.embedder_name}")
    print()

    print("[2/4] Building systems...")
    from ahrag.eval.systems import build_systems

    all_systems = build_systems(
        engine.settings, engine.config, engine.db, today=engine.freshness.today
    )
    if args.systems:
        wanted = {k.strip() for k in args.systems.split(",")}
        systems = [s for s in all_systems if s.key in wanted]
        if not systems:
            print(f"  no systems matched {sorted(wanted)}; "
                  f"available: {[s.key for s in all_systems]}")
            sys.exit(1)
    else:
        systems = all_systems
    print(f"  {', '.join(s.key for s in systems)}")
    print()

    print("[3/4] Running evaluation...")
    results = collect_rows(engine, systems, items)
    print()

    print("[4/4] Statistics")
    print()
    print("=" * 100)
    print("PER-SYSTEM RESULTS (mean [95% bootstrap CI])")
    print("=" * 100)
    shown = METRICS[:5]
    print(f"  {'sys':5s}" + "".join(f"{label[:15]:>18s}" for _, label, _ in shown))
    print("-" * 100)

    vectors_by_metric = {
        metric: aligned_vectors(results, metric) for metric, _, _ in METRICS
    }
    for system in systems:
        cells = []
        for metric, _, _ in shown:
            values = vectors_by_metric[metric].get(system.key)
            if values is None or values.size == 0:
                cells.append(f"{'n/a':>18s}")
                continue
            mean, low, high = bootstrap_ci(values)
            cells.append(f"{mean:.3f} [{low:.2f},{high:.2f}]".rjust(18))
        print(f"  {system.key:5s}" + "".join(cells))
    print()
    n_used = next(
        (v[systems[0].key].size for v in vectors_by_metric.values()
         if systems[0].key in v and v[systems[0].key].size),
        0,
    )
    print(f"  Aligned on {n_used} items where every system produced a value "
          f"(of {len(items)} total).")
    print()

    proposed = args.proposed if any(s.key == args.proposed for s in systems) else systems[-1].key
    print("=" * 100)
    print(f"PAIRWISE COMPARISONS versus {proposed}")
    print("=" * 100)
    print(f"  {'vs':6s} {'metric':20s} {proposed:>8s} {'other':>8s} "
          f"{'delta':>9s} {'p':>8s} {'d':>7s} {'':4s} effect")
    print("-" * 100)

    comparisons: dict[str, dict] = {}
    for system in systems:
        if system.key == proposed:
            continue
        comparisons[system.key] = {}
        for metric, label, higher_better in METRICS:
            vectors = vectors_by_metric[metric]
            if proposed not in vectors or system.key not in vectors:
                continue
            treatment = vectors[proposed]
            control = vectors[system.key]
            if treatment.size == 0:
                continue
            delta, p_value = paired_bootstrap(treatment, control)
            effect = cohens_d(treatment, control)
            comparisons[system.key][metric] = {
                "proposed_mean": float(treatment.mean()),
                "other_mean": float(control.mean()),
                "delta": delta,
                "p_value": p_value,
                "cohens_d": effect,
                "effect_size": interpret_d(effect),
                "higher_is_better": higher_better,
                "significant_at_05": bool(p_value < 0.05),
            }
            print(f"  {system.key:6s} {label:20s} {treatment.mean():8.3f} "
                  f"{control.mean():8.3f} {delta:+9.4f} {p_value:8.4f} "
                  f"{effect:+7.3f} {significance_marker(p_value):4s} "
                  f"{interpret_d(effect)}")
        print()

    print("=" * 100)
    print("ROUTE DISTRIBUTION, EFFICIENCY, AND ABSTENTION")
    print("=" * 100)
    print(f"  {'sys':5s} {'routes':40s} {'latency':>10s} {'cost':>11s} {'abstain':>9s}")
    print("-" * 100)
    efficiency: dict[str, dict] = {}
    for system in systems:
        rows = results[system.key]
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row.get("route", "?")] += 1
        latency = float(np.mean([r.get("latency_s") or 0.0 for r in rows]))
        cost = float(np.mean([r.get("estimated_cost_usd") or 0.0 for r in rows]))
        abstained = sum(1 for r in rows if r.get("abstained")) / max(1, len(rows))
        efficiency[system.key] = {
            "route_distribution": dict(sorted(counts.items())),
            "mean_latency_s": latency,
            "mean_cost_usd": cost,
            "abstention_rate": abstained,
        }
        route_text = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"  {system.key:5s} {route_text:40s} {latency:9.4f}s "
              f"${cost:.6f} {abstained:9.3f}")
    print()

    per_type: dict[str, dict] = {}
    if args.by_type:
        print("=" * 100)
        print("RECALL@5 BY QUERY TYPE (stratified, §2d)")
        print("=" * 100)
        types = sorted({item.query_type for item in items})
        header = f"  {'query type':34s}" + "".join(f"{s.key:>8s}" for s in systems)
        print(header + f"{'n':>6s}")
        print("-" * 100)
        for query_type in types:
            indices = [i for i, item in enumerate(items) if item.query_type == query_type]
            row_cells = []
            per_type[query_type] = {}
            count = 0
            for system in systems:
                rows = results[system.key]
                values = [
                    rows[i]["recall_at_5"] for i in indices
                    if rows[i]["recall_at_5"] is not None
                ]
                count = max(count, len(values))
                if values:
                    mean = float(np.mean(values))
                    per_type[query_type][system.key] = mean
                    row_cells.append(f"{mean:8.3f}")
                else:
                    row_cells.append(f"{'—':>8s}")
            print(f"  {query_type:34s}" + "".join(row_cells) + f"{count:6d}")
        print()

    if args.output:
        payload = {
            "corpus": str(manifest) if manifest else "seed",
            "eval_set": str(eval_set) if eval_set else "seed",
            "label_report": label_report,
            "embedder": engine.index.embedder_name,
            "items": len(items),
            "aligned_items": int(n_used),
            "proposed": proposed,
            "systems": {
                s.key: {"name": s.name, "description": s.description} for s in systems
            },
            "per_system": {
                s.key: {
                    metric: dict(
                        zip(
                            ("mean", "ci_low", "ci_high"),
                            bootstrap_ci(vectors_by_metric[metric][s.key]),
                        )
                    )
                    for metric, _, _ in METRICS
                    if s.key in vectors_by_metric[metric]
                    and vectors_by_metric[metric][s.key].size
                }
                for s in systems
            },
            "comparisons": comparisons,
            "efficiency": efficiency,
            "recall_by_query_type": per_type,
        }
        Path(args.output).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
