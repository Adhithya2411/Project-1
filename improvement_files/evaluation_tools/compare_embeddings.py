"""Compare embedding backends on the AHRAG evaluation suite (improvement.txt §4).

§4 is blunt about this being unfinished work: "sentence-transformers is optional
but never actually evaluated… This is low-hanging fruit." It asks three
questions, and this script answers each with a separate table.

Q1  Does a neural encoder beat the offline LSA default?
    §4(a)/(b): run the same evaluation across LSA, a small sentence-transformer,
    and a larger one. Everything except the embedding backend is held fixed —
    same corpus, same chunking, same BM25 index, same reranker, same router,
    same evidence gates — so the comparison is an ablation of the encoder.

Q2  Does LSA improve with corpus size?
    §4(c): "On a large corpus, LSA will actually learn meaningful term
    associations. Show this empirically: LSA@60 chunks vs LSA@5000 chunks vs
    neural." Run with ``--scaling`` against both corpora.

Q3  Does the encoder change how much routing matters?
    §4(b) again: the R1-vs-R2 decision should matter *more* with a good encoder,
    because dense retrieval then behaves genuinely differently from sparse. This
    is measured directly as the spread between the fixed-route baselines, and as
    sparse/dense rank agreement.

A caveat that applies to every number below: the reranker is pinned to the
lexical backend throughout. Leaving it on ``auto`` would swap in a
cross-encoder alongside the bi-encoder, and the resulting delta could not be
attributed to either.

USAGE
  python improvement_files/evaluation_tools/compare_embeddings.py --integrated
  python improvement_files/evaluation_tools/compare_embeddings.py --integrated \
      --backends lsa,minilm
  python improvement_files/evaluation_tools/compare_embeddings.py --scaling
  python improvement_files/evaluation_tools/compare_embeddings.py --integrated \
      --output embeddings.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ahrag.eval.harness import (  # noqa: E402
    INTEGRATED_EVAL_SET,
    INTEGRATED_MANIFEST,
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

# (key, label, Settings overrides). The two neural models are the ones §4(b)
# names explicitly.
BACKENDS: dict[str, tuple[str, dict]] = {
    "hashing": (
        "Hashing n-grams (floor)",
        {"embedding_backend": "hashing"},
    ),
    "lsa": (
        "LSA / TF-IDF + SVD (offline default)",
        {"embedding_backend": "lsa"},
    ),
    "minilm": (
        "all-MiniLM-L6-v2 (neural, 384d)",
        {
            "embedding_backend": "sentence-transformers",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    ),
    "mpnet": (
        "all-mpnet-base-v2 (neural, 768d)",
        {
            "embedding_backend": "sentence-transformers",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        },
    ),
    "bge": (
        "BGE-base-en-v1.5 (neural, 768d)",
        {
            "embedding_backend": "sentence-transformers",
            "embedding_model": "BAAI/bge-base-en-v1.5",
        },
    ),
}

# Per-row keys from ``ahrag.evaluate.run_item``, not the aggregate names.
METRICS = [
    ("recall_at_5", "Recall@5"),
    ("mrr", "MRR"),
    ("ndcg_at_10", "nDCG@10"),
    ("abstention_appropriate", "Abstention-ok"),
]


def score_system(engine, items) -> tuple[dict[str, np.ndarray], dict]:
    """Run the evaluation suite and return per-item metric vectors.

    Callers that need a different routing policy construct the engine
    themselves (see :func:`route_spread`); this function does not rebuild one,
    so the already-fitted encoder is always the one being measured.
    """
    from ahrag.eval.systems import SystemSpec
    from ahrag.evaluate import run_item

    spec = SystemSpec(key="X", name="x", description="", engine=engine)
    rows = [run_item(spec, item, engine) for item in items]

    vectors: dict[str, np.ndarray] = {}
    mask = np.array([r.get("recall_at_5") is not None for r in rows], dtype=bool)
    for key, _ in METRICS:
        values = [r.get(key) for r in rows]
        if key == "abstention_appropriate":
            vectors[key] = np.array(
                [float(v) for v in values if v is not None], dtype=np.float64
            )
        else:
            vectors[key] = np.array(
                [float(v) for v in np.array(values, dtype=object)[mask]],
                dtype=np.float64,
            )

    routes: dict[str, int] = {}
    for row in rows:
        routes[row.get("route", "?")] = routes.get(row.get("route", "?"), 0) + 1
    summary = {
        "route_distribution": dict(sorted(routes.items())),
        "errors": sum(1 for r in rows if r.get("error")),
        "mean_latency_s": float(
            np.mean([r.get("latency_s") or 0.0 for r in rows])
        ),
    }
    return vectors, summary


def probe_agreement_stats(engine, items) -> dict[str, float]:
    """How differently do sparse and dense rank the same authorised pool?

    This is the Q3 measurement. When the encoder is weak, dense and sparse
    return near-identical orderings and the router's R1-versus-R2 choice is
    close to a coin flip about nothing. A good encoder should *lower* agreement,
    which is what makes the choice consequential.
    """
    agreements: list[float] = []
    dense_only: list[float] = []
    for item in items:
        try:
            user = engine.get_user(item.user_id)
        except KeyError:
            continue
        scope = engine.acl.scope_for(user)
        if not scope.allowed_chunk_ids:
            continue
        probe = engine.retrieval.probe(item.query, scope)
        sparse = set(probe.sparse_ids)
        dense = set(probe.dense_ids)
        union = sparse | dense
        if union:
            agreements.append(len(sparse & dense) / len(union))
            dense_only.append(len(dense - sparse) / len(union))
    return {
        "mean_sparse_dense_jaccard": float(np.mean(agreements)) if agreements else 0.0,
        "mean_dense_only_fraction": float(np.mean(dense_only)) if dense_only else 0.0,
        "n": len(agreements),
    }


def route_spread(engine, items) -> dict[str, float]:
    """Recall@5 for each fixed route, and the spread between them.

    A large spread means the route choice has consequences on this corpus with
    this encoder; a spread near zero means every route finds everything and no
    router can add value, which is precisely improvement.txt §1's complaint
    about the demo corpus.
    """
    from ahrag.models import Route
    from ahrag.pipeline import AHRAGEngine
    from ahrag.routing.router import FixedRouter

    results: dict[str, float] = {}
    for route in (Route.R1, Route.R2, Route.R3, Route.R4):
        forced = AHRAGEngine(
            settings=engine.settings,
            config=engine.config,
            db=engine.db,
            router=FixedRouter(route),
            today=engine.freshness.today,
        )
        forced.index = engine.index
        forced.acl = engine.acl
        forced.retrieval.index = engine.index
        forced.retrieval.acl = engine.acl
        forced.validator.acl = engine.acl
        vectors, _ = score_system(forced, items)
        values = vectors["recall_at_5"]
        results[route.value] = float(values.mean()) if values.size else 0.0

    spread = max(results.values()) - min(results.values()) if results else 0.0
    results["spread"] = spread
    return results


def run_backend(key: str, manifest, eval_set, rebuild: bool):
    """Build an engine with one backend and score it."""
    label, overrides = BACKENDS[key]
    settings_overrides = dict(overrides)
    # Pin the reranker so an encoder change is not confounded with a
    # cross-encoder appearing alongside it.
    settings_overrides["reranker"] = "lexical"

    print(f"  [{key}] {label}")
    started = time.perf_counter()
    try:
        engine = build_seeded_engine(
            manifest,
            settings_overrides=settings_overrides,
            rebuild_cache=rebuild,
            quiet=True,
        )
    except (ImportError, RuntimeError) as exc:
        print(f"      unavailable: {exc}")
        return None
    items, _ = load_and_validate(engine, eval_set, quiet=True)

    vectors, summary = score_system(engine, items)
    agreement = probe_agreement_stats(engine, items)
    elapsed = time.perf_counter() - started

    recall = vectors["recall_at_5"]
    print(
        f"      R@5={recall.mean():.4f}  dim={getattr(engine.index.embedder, 'dim', '?')}  "
        f"sparse/dense Jaccard={agreement['mean_sparse_dense_jaccard']:.3f}  "
        f"({elapsed:.1f}s)"
    )
    return {
        "key": key,
        "label": label,
        "embedder": engine.index.embedder_name,
        "dim": int(getattr(engine.index.embedder, "dim", 0) or 0),
        "vectors": vectors,
        "summary": summary,
        "agreement": agreement,
        "engine": engine,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare AHRAG embedding backends")
    add_corpus_arguments(parser)
    parser.add_argument(
        "--backends",
        type=str,
        default="lsa,minilm,mpnet",
        help=f"Comma-separated. Available: {','.join(BACKENDS)}",
    )
    parser.add_argument(
        "--baseline", type=str, default="lsa",
        help="Backend used as the comparison point for significance tests",
    )
    parser.add_argument(
        "--scaling", action="store_true",
        help="Also run LSA on the seed corpus, to answer §4(c) directly",
    )
    parser.add_argument(
        "--route-spread", action="store_true",
        help="Measure per-fixed-route recall spread per backend (§4b, slower)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    manifest, eval_set = resolve_corpus(args)
    requested = [k.strip() for k in args.backends.split(",") if k.strip()]
    unknown = [k for k in requested if k not in BACKENDS]
    if unknown:
        print(f"Unknown backend(s): {unknown}. Available: {list(BACKENDS)}")
        sys.exit(1)

    print("=" * 92)
    print("EMBEDDING BACKEND COMPARISON")
    print("=" * 92)
    corpus_label = manifest.name if manifest else "packaged seed corpus"
    print(f"corpus: {corpus_label}")
    print("Everything but the embedding backend is held fixed; the reranker is")
    print("pinned to lexical so an encoder change is not confounded with it.")
    print()

    results: dict[str, dict] = {}
    for key in requested:
        outcome = run_backend(key, manifest, eval_set, args.rebuild_cache)
        if outcome is not None:
            if args.limit:
                outcome["items"] = outcome["items"][: args.limit]
            results[key] = outcome
    print()

    if not results:
        print("No backend could be evaluated.")
        sys.exit(1)

    # ---- Q1: side-by-side quality ----
    print("=" * 92)
    print("Q1  RETRIEVAL QUALITY BY BACKEND (mean [95% bootstrap CI])")
    print("=" * 92)
    print(f"  {'backend':38s}" + "".join(f"{label[:15]:>17s}" for _, label in METRICS))
    print("-" * 92)
    for key, entry in results.items():
        cells = []
        for metric, _ in METRICS:
            values = entry["vectors"][metric]
            if values.size == 0:
                cells.append(f"{'n/a':>17s}")
                continue
            mean, low, high = bootstrap_ci(values)
            cells.append(f"{mean:.3f} [{low:.2f},{high:.2f}]".rjust(17))
        print(f"  {entry['label']:<38.36s}" + "".join(cells))
    print()

    baseline_key = args.baseline if args.baseline in results else requested[0]
    baseline = results[baseline_key]
    print(f"  Significance versus {baseline['label']}:")
    print(f"  {'backend':24s} {'metric':16s} {'delta':>9s} {'p':>8s} {'d':>7s} {'':4s} effect")
    print("-" * 92)
    comparisons: dict[str, dict] = {}
    for key, entry in results.items():
        if key == baseline_key:
            continue
        comparisons[key] = {}
        for metric, label in METRICS:
            treatment = entry["vectors"][metric]
            control = baseline["vectors"][metric]
            if treatment.size == 0 or treatment.size != control.size:
                continue
            delta, p_value = paired_bootstrap(treatment, control)
            effect = cohens_d(treatment, control)
            comparisons[key][metric] = {
                "delta": delta,
                "p_value": p_value,
                "cohens_d": effect,
                "effect_size": interpret_d(effect),
            }
            print(f"  {key:24s} {label:16s} {delta:+9.4f} {p_value:8.4f} "
                  f"{effect:+7.3f} {significance_marker(p_value):4s} {interpret_d(effect)}")
        print()

    # ---- Q3: does the encoder make routing matter more? ----
    print("=" * 92)
    print("Q3  DOES THE ENCODER MAKE THE ROUTE CHOICE MATTER MORE?")
    print("=" * 92)
    print("Lower sparse/dense agreement means dense retrieval is contributing")
    print("something sparse does not, which is what makes R1-vs-R2 a real decision.")
    print()
    print(f"  {'backend':38s} {'dim':>6s} {'sparse/dense Jaccard':>22s} {'dense-only share':>18s}")
    print("-" * 92)
    for key, entry in results.items():
        agreement = entry["agreement"]
        print(f"  {entry['label']:<38.36s} {entry['dim']:6d} "
              f"{agreement['mean_sparse_dense_jaccard']:22.4f} "
              f"{agreement['mean_dense_only_fraction']:18.4f}")
    print()

    spreads: dict[str, dict] = {}
    if args.route_spread:
        print(f"  Fixed-route Recall@5 per backend (spread = does the route matter?):")
        print(f"  {'backend':38s} {'R1':>8s} {'R2':>8s} {'R3':>8s} {'R4':>8s} {'spread':>9s}")
        print("-" * 92)
        for key, entry in results.items():
            spread = route_spread(entry["engine"], entry["items"])
            spreads[key] = spread
            print(f"  {entry['label']:<38.36s} {spread['R1']:8.3f} {spread['R2']:8.3f} "
                  f"{spread['R3']:8.3f} {spread['R4']:8.3f} {spread['spread']:9.3f}")
        print()

    # ---- Q2: LSA scaling ----
    scaling: dict[str, dict] = {}
    if args.scaling:
        print("=" * 92)
        print("Q2  DOES LSA IMPROVE WITH CORPUS SIZE? (§4c)")
        print("=" * 92)
        for label, corpus, evaluation in (
            ("seed corpus", None, None),
            ("integrated corpus", INTEGRATED_MANIFEST, INTEGRATED_EVAL_SET),
        ):
            if corpus is not None and not corpus.exists():
                continue
            engine = build_seeded_engine(
                corpus,
                settings_overrides={"embedding_backend": "lsa", "reranker": "lexical"},
                quiet=True,
            )
            items, _ = load_and_validate(engine, evaluation, quiet=True)
            vectors, _ = score_system(engine, items)
            recall = vectors["recall_at_5"]
            mean, low, high = bootstrap_ci(recall)
            agreement = probe_agreement_stats(engine, items)
            scaling[label] = {
                "chunks": engine.db.count_chunks(),
                "items": len(items),
                "lsa_dim": int(getattr(engine.index.embedder, "dim", 0) or 0),
                "recall_at_5": mean,
                "ci95": [low, high],
                "sparse_dense_jaccard": agreement["mean_sparse_dense_jaccard"],
            }
            print(f"  LSA @ {label:22s} {engine.db.count_chunks():6d} chunks  "
                  f"rank={scaling[label]['lsa_dim']:4d}  "
                  f"R@5={mean:.4f} [{low:.2f},{high:.2f}]  "
                  f"(n={len(items)} items)")
        print()
        print("  Note: the two rows use different evaluation sets, so the recall")
        print("  figures are not directly comparable as a single trend. What is")
        print("  comparable is the fitted rank and the sparse/dense divergence,")
        print("  which show whether the SVD had enough co-occurrence to learn from.")
        print()

    if args.output:
        payload = {
            "corpus": str(manifest) if manifest else "seed",
            "baseline": baseline_key,
            "backends": {
                key: {
                    "label": entry["label"],
                    "embedder": entry["embedder"],
                    "dim": entry["dim"],
                    "metrics": {
                        metric: dict(
                            zip(
                                ("mean", "ci_low", "ci_high"),
                                bootstrap_ci(entry["vectors"][metric]),
                            )
                        )
                        for metric, _ in METRICS
                        if entry["vectors"][metric].size
                    },
                    "agreement": entry["agreement"],
                    "summary": entry["summary"],
                }
                for key, entry in results.items()
            },
            "comparisons_vs_baseline": comparisons,
            "route_spread": spreads,
            "lsa_scaling": scaling,
        }
        Path(args.output).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
