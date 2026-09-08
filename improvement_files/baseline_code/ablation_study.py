"""Component ablations for AHRAG (improvement.txt §5c).

The existing evaluation varies only the routing *policy* (B1-B5 versus P1),
which answers "does adaptive routing help?" but not "which part of the
governance machinery is doing the work?". §5c asks for the second question, and
requires each ablation to show a measurable, significant difference — including
when it does not.

Each ablation disables exactly one mechanism by editing the loaded
``RouterConfig`` in memory, so the corpus, indexes, embeddings, reranker,
generator, and ACL enforcement are held identical. Nothing is disabled by
patching Python: every knob here is a policy value that ``config/router.yaml``
already exposes, which is why the file is loaded rather than hard-coded.

One mechanism is deliberately *not* ablated: the ACL pre-filter. It is upstream
of routing and cannot be switched off through configuration — that is the point
of ``INVENTION_DISCLOSURE.md`` M1. ``tests/test_governance.py`` covers it by
asserting that bypassing it raises.

USAGE
  python improvement_files/baseline_code/ablation_study.py --integrated
  python improvement_files/baseline_code/ablation_study.py --integrated --output ablations.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ahrag.eval.harness import (  # noqa: E402
    add_corpus_arguments,
    build_seeded_engine,
    load_and_validate,
    resolve_corpus,
)
from ahrag.stats import bootstrap_ci, cohens_d, paired_bootstrap  # noqa: E402

# Per-row keys as emitted by ``ahrag.evaluate.run_item``. These are NOT the
# aggregate names used by ``ahrag.eval.metrics.aggregate`` — e.g. the row key is
# ``abstention_appropriate`` (a bool per query) while the aggregate is
# ``abstention_appropriateness`` (its mean). Reading the aggregate name off a
# row yields None for every item and the metric silently reports "n/a".
METRICS = [
    ("recall_at_5", "Recall@5"),
    ("mrr", "MRR"),
    ("ndcg_at_10", "nDCG@10"),
    ("abstention_appropriate", "Abstention-ok"),
    ("citation_coverage", "Citation coverage"),
    ("freshness_compliant", "Freshness compliance"),
    ("acl_violation", "ACL violation rate"),
]


# ---------------------------------------------------------------------------
# The ablations
# ---------------------------------------------------------------------------


def ablate_probe_confidence(config) -> None:
    """Remove probe evidence from the quality model and the risk model.

    The router still sees the query text; it just stops being told whether the
    *authorised* corpus contains anything relevant. This is the ablation of
    mechanism M3.
    """
    for spec in config.quality_model.values():
        spec.probe = {}
    config.risk_model.low_probe_weight = 0.0
    # The hard probe floor is part of the same mechanism.
    config.constraints.min_probe_for_answering = 0.0


def ablate_freshness_penalty(config) -> None:
    """Stop penalising routes that cannot surface a superseded/current pair."""
    config.risk_model.freshness_penalty = {}


def ablate_risk_model(config) -> None:
    """Drop the risk term from the utility function entirely."""
    config.lambdas.risk = 0.0
    config.risk_model.base = {}
    config.risk_model.freshness_penalty = {}
    config.risk_model.conflict_penalty = {}
    config.risk_model.restricted_scope_weight = 0.0
    config.risk_model.low_probe_weight = 0.0


def ablate_restricted_scope_signal(config) -> None:
    """Remove the governance signal specifically: how much is withheld.

    Narrower than :func:`ablate_risk_model` — the risk term survives, but the
    router no longer knows that this principal is looking at a filtered view.
    """
    config.risk_model.restricted_scope_weight = 0.0
    for spec in config.quality_model.values():
        spec.weights.pop("restricted_fraction", None)


def ablate_conflict_signal(config) -> None:
    """Stop raising risk for routes that cannot disclose a version conflict."""
    config.risk_model.conflict_penalty = {}


def ablate_cost_and_latency(config) -> None:
    """Make the router quality-only: no cost or latency pressure.

    Predicts a shift towards R4 everywhere. If quality does not improve, the
    cost terms are buying efficiency for free, which is the §8(b) claim.
    """
    config.lambdas.cost = 0.0
    config.lambdas.latency = 0.0


def ablate_authority_gate(config) -> None:
    """Stop requiring a minimum authority score for policy answers."""
    config.evidence.min_authority_for_policy = 1


def ablate_freshness_enforcement(config) -> None:
    """Stop requiring the current version to lead on freshness-sensitive queries."""
    config.evidence.enforce_current_version_on_freshness = False
    config.evidence.demote_superseded = False
    config.evidence.superseded_score_multiplier = 1.0


def ablate_evidence_gate(config) -> None:
    """Lower the sufficiency thresholds to effectively off.

    Isolates how much of the abstention behaviour comes from the gate rather
    than from routing.
    """
    config.evidence.min_top_score = 0.0
    config.evidence.min_mean_score = 0.0
    config.evidence.min_supporting_chunks = 1
    config.evidence.min_distinct_documents = 1
    config.evidence.diversity_required_intents = []


ABLATIONS: list[tuple[str, str, object]] = [
    ("full", "Full system (no ablation)", None),
    ("no_probe", "− probe confidence (M3)", ablate_probe_confidence),
    ("no_restricted_signal", "− restricted-scope signal (M2)", ablate_restricted_scope_signal),
    ("no_freshness_penalty", "− freshness penalty in routing", ablate_freshness_penalty),
    ("no_conflict_signal", "− conflict signal in routing", ablate_conflict_signal),
    ("no_risk_model", "− risk term entirely", ablate_risk_model),
    ("no_cost_latency", "− cost and latency terms", ablate_cost_and_latency),
    ("no_authority_gate", "− authority gate in evidence packing", ablate_authority_gate),
    ("no_freshness_enforcement", "− freshness enforcement in packing", ablate_freshness_enforcement),
    ("no_evidence_gate", "− evidence sufficiency gate", ablate_evidence_gate),
]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_ablation(base_engine, items, mutate) -> tuple[dict[str, np.ndarray], dict]:
    """Run one ablation and return per-item metric vectors plus a summary."""
    from ahrag.eval.systems import SystemSpec
    from ahrag.evaluate import run_item
    from ahrag.pipeline import AHRAGEngine
    from ahrag.routing.router import GovernanceAwareRouter

    config = copy.deepcopy(base_engine.config)
    if mutate is not None:
        mutate(config)

    engine = AHRAGEngine(
        settings=base_engine.settings,
        config=config,
        db=base_engine.db,
        router=GovernanceAwareRouter(config, base_engine.settings),
        today=base_engine.freshness.today,
    )
    # Share the built index: only policy differs, so refitting the embedding
    # space per ablation would multiply the cost for no change in results.
    engine.index = base_engine.index
    engine.acl = base_engine.acl
    engine.retrieval.index = base_engine.index
    engine.retrieval.acl = base_engine.acl
    engine.validator.acl = base_engine.acl

    spec = SystemSpec(key="ABL", name="ablation", description="", engine=engine)
    rows = [run_item(spec, item, engine) for item in items]

    vectors: dict[str, np.ndarray] = {}
    for key, _ in METRICS:
        values = [r.get(key) for r in rows]
        vectors[key] = np.array(
            [float(v) for v in values if v is not None], dtype=np.float64
        )

    routes: dict[str, int] = {}
    for row in rows:
        route = row.get("route", "?")
        routes[route] = routes.get(route, 0) + 1

    summary = {
        "route_distribution": dict(sorted(routes.items())),
        "mean_latency_s": float(
            np.mean([r.get("latency_s", 0.0) or 0.0 for r in rows])
        ),
        "mean_cost_usd": float(
            np.mean([r.get("estimated_cost_usd", 0.0) or 0.0 for r in rows])
        ),
        "errors": sum(1 for r in rows if r.get("error")),
    }
    return vectors, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="AHRAG component ablations")
    add_corpus_arguments(parser)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of evaluation queries")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated ablation keys to run")
    parser.add_argument("--output", type=str, default=None,
                        help="Write the full report to this JSON path")
    args = parser.parse_args()

    manifest, eval_set = resolve_corpus(args)

    print("=" * 96)
    print("AHRAG COMPONENT ABLATIONS")
    print("=" * 96)
    print("Every row shares one corpus, one index, one generator, and one ACL")
    print("layer. Only the named mechanism is disabled.")
    print()

    engine = build_seeded_engine(manifest, rebuild_cache=args.rebuild_cache)
    items, _ = load_and_validate(engine, eval_set)
    if args.limit:
        items = items[: args.limit]
        print(f"  limited to {len(items)} queries")
    print()

    selected = ABLATIONS
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        selected = [a for a in ABLATIONS if a[0] in wanted or a[0] == "full"]

    results: dict[str, dict] = {}
    for key, label, mutate in selected:
        started = time.perf_counter()
        vectors, summary = run_ablation(engine, items, mutate)
        elapsed = time.perf_counter() - started
        results[key] = {"label": label, "vectors": vectors, "summary": summary}
        headline = vectors.get("recall_at_5")
        recall = f"{headline.mean():.4f}" if headline is not None and headline.size else "n/a"
        print(f"  {key:26s} R@5={recall}  ({elapsed:.1f}s, "
              f"{summary['errors']} errors)")
    print()

    baseline = results.get("full")
    if baseline is None:
        print("The 'full' row is required as the comparison point.")
        sys.exit(1)

    print("=" * 96)
    print("PER-METRIC RESULTS (mean [95% bootstrap CI])")
    print("=" * 96)
    header = f"{'ablation':26s}" + "".join(f"{label[:17]:>19s}" for _, label in METRICS[:4])
    print(header)
    print("-" * 96)
    for key, entry in results.items():
        cells = []
        for metric, _ in METRICS[:4]:
            values = entry["vectors"][metric]
            if values.size == 0:
                cells.append(f"{'n/a':>19s}")
                continue
            mean, low, high = bootstrap_ci(values)
            cells.append(f"{mean:.3f} [{low:.2f},{high:.2f}]".rjust(19))
        print(f"  {key:24s}" + "".join(cells))
    print()

    print("=" * 96)
    print("EFFECT OF EACH ABLATION versus the full system")
    print("=" * 96)
    print(f"  {'ablation':26s} {'metric':22s} {'delta':>9s} {'p':>8s} {'d':>7s}  verdict")
    print("-" * 96)

    report: dict[str, dict] = {}
    for key, entry in results.items():
        if key == "full":
            continue
        report[key] = {"label": entry["label"], "summary": entry["summary"], "metrics": {}}
        for metric, label in METRICS:
            ablated = entry["vectors"][metric]
            full = baseline["vectors"][metric]
            if ablated.size == 0 or ablated.size != full.size:
                continue
            delta, p_value = paired_bootstrap(ablated, full)
            effect = cohens_d(ablated, full)
            if p_value < 0.05:
                verdict = "significant"
            elif abs(delta) < 1e-9:
                verdict = "no change"
            else:
                verdict = "not significant"
            report[key]["metrics"][metric] = {
                "delta_vs_full": delta,
                "p_value": p_value,
                "cohens_d": effect,
                "ablated_mean": float(ablated.mean()),
                "full_mean": float(full.mean()),
            }
            print(f"  {key:26s} {label:22s} {delta:+9.4f} {p_value:8.4f} "
                  f"{effect:+7.3f}  {verdict}")
        print()

    print("=" * 96)
    print("ROUTE DISTRIBUTION AND EFFICIENCY")
    print("=" * 96)
    print(f"  {'ablation':26s} {'routes':44s} {'latency':>10s} {'cost':>11s}")
    print("-" * 96)
    for key, entry in results.items():
        summary = entry["summary"]
        routes = " ".join(f"{k}:{v}" for k, v in summary["route_distribution"].items())
        print(f"  {key:26s} {routes:44s} {summary['mean_latency_s']:9.4f}s "
              f"${summary['mean_cost_usd']:.6f}")
    print()

    if args.output:
        payload = {
            "corpus": str(manifest) if manifest else "seed",
            "eval_set": str(eval_set) if eval_set else "seed",
            "items": len(items),
            "full_system": {
                metric: dict(
                    zip(
                        ("mean", "ci_low", "ci_high"),
                        bootstrap_ci(baseline["vectors"][metric]),
                    )
                )
                for metric, _ in METRICS
                if baseline["vectors"][metric].size
            },
            "ablations": report,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
