"""Measure Scope-Pure Index Specialisation (SPIS).

Three experiments, reported separately because they answer different questions
and — on the current corpus — give different verdicts.

E1. LATTICE STRUCTURE
    Does the corpus have distinct ACL equivalence classes at all? On a corpus
    where every document grants the same role, specialisation is provably a
    no-op and E2/E3 are meaningless. Printed first so that is never a surprise.

E2. INTERFERENCE (the security claim)
    Hold a principal's authorised subcorpus fixed and delete everything it
    cannot read. A non-interfering system must return bit-identical rankings.
    Reported as Kendall tau-b and top-1 flip rate between the two conditions,
    for the unspecialised index and for SPIS. This does not depend on corpus
    size.

E3. RETRIEVAL QUALITY AND THE LAMBDA FRONTIER (the performance claim)
    R@5 / MRR / nDCG@10 against gold labels, unspecialised versus SPIS across
    lambda in [0, 1], with paired bootstrap confidence intervals and Cohen's d.
    lambda=0 is provably pure; lambda=1 restores global *sparse* statistics.
    lambda does not govern the dense channel, so the lambda=1 row is not the
    unspecialised baseline — the separate `unspecialised` row is.

USAGE
  python improvement_files/evaluation_tools/measure_specialisation.py
  python improvement_files/evaluation_tools/measure_specialisation.py --lambdas 0,0.25,0.5,1
  python improvement_files/evaluation_tools/measure_specialisation.py --output spis.json
  # Larger corpus / eval set, once their gold chunk IDs are repaired:
  python improvement_files/evaluation_tools/measure_specialisation.py \
      --manifest improvement_files/datasets/integrated/manifest.yaml \
      --eval-set improvement_files/datasets/integrated/eval_set.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
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

# Fixed reference date so freshness-dependent behaviour does not drift with the
# real clock, matching the rest of the evaluation harness.
EVAL_TODAY = date(2026, 8, 19)

SEED = 1729


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def kendall_tau_b(a: list[str], b: list[str]) -> float | None:
    """Kendall tau-b over the ranking of items common to both lists.

    Returns None when fewer than two items are shared, where correlation is
    undefined rather than zero.
    """
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return None
    rank_b = {cid: i for i, cid in enumerate(b)}
    order = [rank_b[cid] for cid in common]
    n = len(order)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign = (order[j] - order[i])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------


def build_engine(
    tmp_dir: Path,
    manifest: Path | None,
    specialised: bool,
    lam: float = 0.0,
    calibrate: bool = False,
    isolated: bool = False,
):
    """Build an engine with the given SPIS settings.

    Args:
        tmp_dir: Retained for the isolated path, where each engine needs its
            own database because the corpus itself is mutated.
        manifest: Corpus manifest, or None for the packaged seed corpus.
        specialised: Enable scope-pure index specialisation.
        lam: Information-flow budget for the sparse channel.
        calibrate: Enable per-class confidence calibration.
        isolated: Seed a throwaway database instead of using the shared cache.
            Required by E2, which deletes documents from the corpus; the
            cached corpus must never be mutated.
    """
    overrides = {
        "index_specialisation": specialised,
        "specialisation_lambda": lam,
        "specialisation_calibrate_confidence": calibrate,
    }
    if isolated:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        from ahrag.config import RouterConfig, Settings
        from ahrag.db import Database
        from ahrag.pipeline import AHRAGEngine

        settings = Settings(
            data_dir=tmp_dir,
            db_path=tmp_dir / "spis.sqlite3",
            router_config=PROJECT_ROOT / "config" / "router.yaml",
            **overrides,
        )
        engine = AHRAGEngine(
            settings=settings,
            config=RouterConfig.load(settings.router_config),
            db=Database(settings.db_path),
            today=EVAL_TODAY,
        )
        engine.seed(reset=True, manifest_path=manifest)
        return engine

    return build_seeded_engine(
        manifest, settings_overrides=overrides, today=EVAL_TODAY, quiet=True
    )


def prune_to_scope(engine, user_id: str) -> None:
    """Delete every document ``user_id`` cannot read, then rebuild indexes."""
    user = engine.get_user(user_id)
    for document in list(engine.db.get_documents()):
        if not any(role.lower() in user.role_set for role in document.acl_roles):
            engine.db.delete_document(document.doc_id)
    engine.refresh_indexes()


# ---------------------------------------------------------------------------
# E1 — lattice structure
# ---------------------------------------------------------------------------


def experiment_lattice(engine) -> dict:
    """Report the ACL lattice induced by the known principals."""
    from ahrag.index.lattice import ScopeLattice

    lattice = ScopeLattice(engine.db.get_chunks())
    scopes = [engine.acl.scope_for(u) for u in engine.db.get_users()]
    summary = lattice.describe(scopes)

    print("=" * 78)
    print("E1  ACL LATTICE STRUCTURE")
    print("=" * 78)
    print(f"  corpus                : {len(engine.db.get_documents())} docs, "
          f"{summary['total_chunks']} chunks")
    print(f"  principals            : {summary['principals']}")
    print(f"  distinct ACL classes  : {summary['distinct_classes']}")
    print(f"  single-role classes   : {summary['role_classes']}")
    print(f"  class sizes (chunks)  : {summary['class_sizes']}")
    print(f"  smallest / total      : {summary['smallest_class_fraction']:.4f}")
    if summary["specialisation_is_degenerate"]:
        print()
        print("  *** DEGENERATE: every principal has the same authorised scope. ***")
        print("  *** Specialisation is provably a no-op here; E2/E3 are vacuous. ***")
    print()
    for signature, members in sorted(summary["members"].items()):
        scope = next(s for s in scopes if s.user_id == members[0])
        print(f"    {signature[:12]}  {len(scope.allowed_chunk_ids):5d} chunks  {members}")
    print()
    return summary


# ---------------------------------------------------------------------------
# E2 — interference
# ---------------------------------------------------------------------------


def experiment_interference(
    tmp_root: Path, manifest: Path | None, queries: list[str], observers: list[str]
) -> dict:
    """Measure whether unreadable documents change an authorised user's results."""
    print("=" * 78)
    print("E2  INTERFERENCE FROM UNREADABLE DOCUMENTS")
    print("=" * 78)
    print("  Condition A: full corpus.  Condition B: unreadable documents deleted.")
    print("  A non-interfering system returns identical rankings. tau=1.000, flips=0.")
    print()

    results: dict[str, dict] = {}
    for label, specialised in (("unspecialised", False), ("SPIS (lambda=0)", True)):
        taus: list[float] = []
        flips = 0
        order_changes = 0
        route_changes = 0
        comparisons = 0

        for observer in observers:
            full = build_engine(
                tmp_root / f"{int(specialised)}-full-{observer}", manifest,
                specialised, isolated=True,
            )
            pruned = build_engine(
                tmp_root / f"{int(specialised)}-pruned-{observer}", manifest,
                specialised, isolated=True,
            )
            prune_to_scope(pruned, observer)
            if pruned.db.count_chunks() >= full.db.count_chunks():
                continue

            for query in queries:
                a = full.answer(query, observer, write_audit=False)
                b = pruned.answer(query, observer, write_audit=False)
                ra = [e.chunk_id for e in a.evidence]
                rb = [e.chunk_id for e in b.evidence]
                comparisons += 1
                tau = kendall_tau_b(ra, rb)
                if tau is not None:
                    taus.append(tau)
                if ra[:1] != rb[:1]:
                    flips += 1
                if ra != rb:
                    order_changes += 1
                if a.decision.route is not b.decision.route:
                    route_changes += 1

        mean_tau = float(np.mean(taus)) if taus else float("nan")
        results[label] = {
            "comparisons": comparisons,
            "mean_kendall_tau": mean_tau,
            "top1_flip_rate": flips / comparisons if comparisons else 0.0,
            "order_change_rate": order_changes / comparisons if comparisons else 0.0,
            "route_change_rate": route_changes / comparisons if comparisons else 0.0,
            "non_interfering": order_changes == 0 and route_changes == 0,
        }
        r = results[label]
        print(f"  {label:18s} n={comparisons:4d}  tau={mean_tau:6.3f}  "
              f"top1-flips={r['top1_flip_rate']:.3f}  "
              f"order-changes={r['order_change_rate']:.3f}  "
              f"route-changes={r['route_change_rate']:.3f}  "
              f"NON-INTERFERING={r['non_interfering']}")
    print()
    return results


# ---------------------------------------------------------------------------
# E3 — retrieval quality and the lambda frontier
# ---------------------------------------------------------------------------


def score_engine(engine, items) -> dict[str, np.ndarray]:
    """Run the eval set and return per-item metric vectors."""
    from ahrag.eval.metrics import (
        abstention_appropriate,
        mrr,
        ndcg_at_k,
        recall_at_k,
    )

    recall: list[float] = []
    reciprocal: list[float] = []
    ndcg: list[float] = []
    abstention: list[float] = []

    for item in items:
        result = engine.answer(item.query, item.user_id, write_audit=False)
        retrieved = [e.chunk_id for e in result.evidence]
        abstention.append(
            1.0 if abstention_appropriate(item.should_abstain, result.abstained) else 0.0
        )
        if not item.gold_chunks:
            continue
        gold = item.gold_set
        recall.append(recall_at_k(retrieved, gold, 5) or 0.0)
        reciprocal.append(mrr(retrieved, gold) or 0.0)
        ndcg.append(ndcg_at_k(retrieved, gold, 10) or 0.0)

    return {
        "recall_at_5": np.array(recall),
        "mrr": np.array(reciprocal),
        "ndcg_at_10": np.array(ndcg),
        "abstention_appropriateness": np.array(abstention),
    }


def experiment_quality(
    tmp_root: Path, manifest: Path | None, eval_set: Path | None, lambdas: list[float]
) -> dict:
    """Compare retrieval quality across the purity/utility frontier."""
    reference = build_engine(tmp_root / "q-load", manifest, specialised=False)
    items, _ = load_and_validate(reference, eval_set, quiet=True)
    answerable = sum(1 for i in items if i.gold_chunks)

    print("=" * 78)
    print("E3  RETRIEVAL QUALITY AND THE LAMBDA FRONTIER")
    print("=" * 78)
    print(f"  {len(items)} eval items ({answerable} answerable)")
    print()

    baseline = score_engine(reference, items)

    metrics = ("recall_at_5", "mrr", "ndcg_at_10", "abstention_appropriateness")
    print(f"  {'system':18s} " + " ".join(f"{m[:12]:>14s}" for m in metrics))
    print("  " + "-" * 76)

    def row(label: str, scores: dict[str, np.ndarray]) -> None:
        cells = []
        for metric in metrics:
            mean, lo, hi = bootstrap_ci(scores[metric])
            cells.append(f"{mean:.3f} [{lo:.2f},{hi:.2f}]".rjust(14))
        print(f"  {label:18s} " + " ".join(cells))

    row("unspecialised", baseline)

    frontier: dict[str, dict] = {}
    for lam in lambdas:
        engine = build_engine(
            tmp_root / f"q-lam{lam}", manifest, specialised=True, lam=lam
        )
        scores = score_engine(engine, items)
        row(f"SPIS lambda={lam:g}", scores)
        entry: dict[str, object] = {
            "provably_pure": lam == 0.0,
            "specialisation_stats": engine.index.specialisation.stats(),
        }
        for metric in metrics:
            mean, lo, hi = bootstrap_ci(scores[metric])
            delta, p_value = paired_bootstrap(scores[metric], baseline[metric])
            effect = cohens_d(scores[metric], baseline[metric])
            entry[metric] = {
                "mean": mean,
                "ci95": [lo, hi],
                "delta": delta,
                "p_value": p_value,
                "cohens_d": effect,
            }
        frontier[f"lambda={lam:g}"] = entry

    print()
    print(f"  {'comparison':18s} {'metric':28s} {'delta':>8s} {'p':>8s} {'d':>7s}  verdict")
    print("  " + "-" * 76)
    for key, entry in frontier.items():
        for metric in metrics:
            stat = entry[metric]
            verdict = (
                "significant" if stat["p_value"] < 0.05 else "not significant"
            )
            print(f"  {key:18s} {metric:28s} {stat['delta']:+8.4f} "
                  f"{stat['p_value']:8.4f} {stat['cohens_d']:+7.3f}  {verdict}")
    print()

    return {
        "eval_items": len(items),
        "answerable_items": answerable,
        "baseline": {
            m: dict(zip(("mean", "ci_low", "ci_high"), bootstrap_ci(baseline[m])))
            for m in metrics
        },
        "frontier": frontier,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Scope-Pure Index Specialisation",
    )
    add_corpus_arguments(parser)
    parser.add_argument("--lambdas", type=str, default="0,0.25,0.5,0.75,1",
                        help="Comma-separated information-flow budgets to sweep")
    parser.add_argument("--observers", type=str,
                        default="erin.contractor,dan.finance,alice.employee",
                        help="Principals used as interference observers")
    parser.add_argument("--output", type=str, default=None,
                        help="Write the full report to this JSON path")
    parser.add_argument("--skip-interference", action="store_true",
                        help="Skip E2, which reseeds one engine pair per observer")
    args = parser.parse_args()

    manifest, eval_set = resolve_corpus(args)
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    observers = [o.strip() for o in args.observers.split(",") if o.strip()]

    import tempfile

    print()
    print("#" * 78)
    print("#  Scope-Pure Index Specialisation — measurement report")
    print("#" * 78)
    print()

    with tempfile.TemporaryDirectory(prefix="spis-") as tmp:
        tmp_root = Path(tmp)
        engine = build_engine(tmp_root / "lattice", manifest, specialised=True)
        report: dict[str, object] = {
            "manifest": str(manifest) if manifest else "seed",
            "eval_set": str(eval_set) if eval_set else "seed",
            "lattice": experiment_lattice(engine),
        }

        if not args.skip_interference:
            report["interference"] = experiment_interference(
                tmp_root, manifest, INTERFERENCE_QUERIES, observers
            )

        report["quality"] = experiment_quality(tmp_root, manifest, eval_set, lambdas)

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Report written to {out}")


INTERFERENCE_QUERIES = [
    "how many annual leave days do I get",
    "what is the expense approval limit for travel",
    "remote work eligibility requirements",
    "payment gateway error code retry",
    "quarterly revenue forecast",
    "who approves a policy exception",
    "what is the incident escalation path",
    "how do I request unpaid leave",
]


if __name__ == "__main__":
    main()
