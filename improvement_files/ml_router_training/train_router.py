"""Train a learned route selector for AHRAG.

This addresses improvement.txt §3, whose complaint is that the router's 23
feature weights were hand-set and never fitted to data, so "adaptive" describes
a heuristic rather than a learned policy.

WHAT WAS WRONG WITH THE PREVIOUS VERSION OF THIS SCRIPT
-------------------------------------------------------
It reported 97.2% validation accuracy, and that number meant nothing.

* Its ``generate_gold_labels`` looped ``for route in route_objects:`` but the
  loop body called ``engine.answer(...)``, which ignores ``route`` and lets the
  engine's own router choose. The same result was recomputed four times per
  query. No route was ever actually forced, so every answerable query whose
  evidence touched a gold chunk collapsed to the label ``R1``.
* 500 of its 532 training rows were HotpotQA questions labelled by a rule over
  HotpotQA's own ``type``/``level`` metadata, while the features
  ``comparison_signal`` and ``hop_signal`` are lexical functions of the same
  question text. The classifier was rediscovering the labelling rule.
* Those rows were extracted against an **empty** ``AuthorisedScope``, so every
  probe feature was identically zero and one split on ``sparse_confidence``
  separated them perfectly from the real rows.
* It reported the validation split as its headline figure — no held-out test
  set, contrary to §3(c).
* Its "rule-based router" baseline used ``item.expected_route``, the human
  annotation from ``eval_set.yaml``, as the router's *prediction*.
  ``GovernanceAwareRouter.decide`` was never called, so the comparison measured
  agreement between two label columns.

WHAT THIS VERSION DOES
----------------------
1. Builds one engine per route with ``FixedRouter``, all sharing a single
   database and a single index, and actually executes all five routes on every
   query (§3a).
2. Derives the gold label from measured outcomes: the cheapest route whose
   recall reaches the best recall any route achieved, subject to the evidence
   gate passing. Abstention items are labelled R0 only when the evidence really
   is insufficient. Queries no route can answer are dropped, not defaulted.
3. Splits 60/20/20 by query. Thresholds and hyperparameters are chosen on
   validation only; test is touched once, at the end (§3c).
4. Compares the trained model against the *actual* rule-based router, the
   complexity-only router, an oracle, and a majority-class baseline (§3b).
5. Sweeps the utility lambdas on validation (§3d).
6. Reports a learning curve over training-set size (§3e).

USAGE
  python improvement_files/ml_router_training/train_router.py --integrated
  python improvement_files/ml_router_training/train_router.py --integrated --dry-run
  python improvement_files/ml_router_training/train_router.py          # seed corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

# Quiet and single-threaded: the transformer tokenizer's parallelism warns on
# every fork, and progress bars make the per-route sweep unreadable in a log.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ahrag.eval.harness import (  # noqa: E402
    add_corpus_arguments,
    build_seeded_engine,
    load_and_validate,
    resolve_corpus,
)
from ahrag.eval.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from ahrag.models import Intent, Route  # noqa: E402

ARTIFACT_DIR = PROJECT_ROOT / "improvement_files" / "ml_router_training" / "artifacts"

FEATURE_NAMES = [
    "query_length", "identifier_signal", "numeric_signal",
    "temporal_signal", "lexical_specificity", "semantic_ambiguity",
    "mixed_signal", "comparison_signal", "hop_signal",
    "followup_signal", "unsupported_signal", "conflict_likelihood",
    "restricted_fraction", "intent_lookup", "intent_explanation",
    "intent_comparison", "intent_summarisation", "intent_procedural",
    "intent_temporal", "sparse_confidence", "dense_confidence",
    "probe_agreement", "query_length_tokens",
]

ROUTE_NAMES = ["R0", "R1", "R2", "R3", "R4"]
ROUTE_TO_INDEX = {name: i for i, name in enumerate(ROUTE_NAMES)}
ROUTE_OBJECTS = [Route.R0, Route.R1, Route.R2, Route.R3, Route.R4]

# Cost ordering used to break ties among routes of equal measured quality.
# This is the "cheapest route that clears the bar" rule of §3(a), and it is the
# only place a prior enters the labels.
ROUTE_COST = {"R0": 0, "R1": 1, "R2": 2, "R3": 3, "R4": 4}

SEED = 1729


def features_to_array(features) -> list[float]:
    """Flatten a ``RouterFeatures`` object into the 23-element feature vector."""
    return [
        features.query_length,
        features.identifier_signal,
        features.numeric_signal,
        features.temporal_signal,
        features.lexical_specificity,
        features.semantic_ambiguity,
        features.mixed_signal,
        features.comparison_signal,
        features.hop_signal,
        features.followup_signal,
        features.unsupported_signal,
        features.conflict_likelihood,
        features.restricted_fraction,
        float(features.intent is Intent.LOOKUP),
        float(features.intent is Intent.EXPLANATION),
        float(features.intent is Intent.COMPARISON),
        float(features.intent is Intent.SUMMARISATION),
        float(features.intent is Intent.PROCEDURAL),
        float(features.intent is Intent.TEMPORAL),
        features.sparse_confidence,
        features.dense_confidence,
        features.probe_agreement,
        float(features.query_length_tokens),
    ]


# ===========================================================================
# STEP 1 — actually run all five routes on every query
# ===========================================================================


def build_route_engines(base_engine):
    """One engine per route, sharing the base engine's database and index.

    Sharing matters: constructing an ``AHRAGEngine`` refits the embedding space,
    which on a 7,000-chunk corpus is the dominant cost. Five independent engines
    would refit five times for no benefit, since only the routing policy differs.
    """
    from ahrag.pipeline import AHRAGEngine
    from ahrag.routing.router import FixedRouter

    engines: dict[str, object] = {}
    for route in ROUTE_OBJECTS:
        engine = AHRAGEngine(
            settings=base_engine.settings,
            config=base_engine.config,
            db=base_engine.db,
            router=FixedRouter(route, name=f"fixed-{route.value.lower()}"),
            today=base_engine.freshness.today,
        )
        # Reuse the already-built index and ACL snapshot rather than rebuilding.
        engine.index = base_engine.index
        engine.acl = base_engine.acl
        engine.retrieval.index = base_engine.index
        engine.retrieval.acl = base_engine.acl
        engine.validator.acl = base_engine.acl
        engines[route.value] = engine
    return engines


def route_records_cache_path(manifest, item_count: int) -> Path:
    """Where the route-execution sweep is cached.

    The sweep costs several minutes and its result depends only on the corpus,
    the evaluation set, and the embedding backend — none of which change while
    a model is being retrained. Caching it makes the Adaptive-RAG baseline and
    the ablations reuse one sweep instead of repeating it.
    """
    import hashlib

    key = hashlib.blake2b(digest_size=8)
    key.update(str(manifest or "seed").encode("utf-8"))
    key.update(str(item_count).encode("utf-8"))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR / f"route_records-{key.hexdigest()}.json"


def run_all_routes(base_engine, items, verbose: bool = True) -> list[dict]:
    """Execute every route on every query and record measured outcomes.

    Returns one record per evaluation item containing the 23 features (extracted
    once, since they do not depend on the route) plus, for each route, the
    recall, nDCG, evidence-gate verdict, abstention flag and latency.
    """
    engines = build_route_engines(base_engine)
    records: list[dict] = []
    started = time.perf_counter()

    for index, item in enumerate(items, start=1):
        try:
            user = base_engine.get_user(item.user_id)
        except KeyError:
            continue

        scope = base_engine.acl.scope_for(user)
        probe = base_engine.retrieval.probe(item.query, scope)
        features = base_engine.features.extract(item.query, user, scope, probe=probe)

        record: dict = {
            "id": item.id,
            "query": item.query,
            "user_id": item.user_id,
            "query_type": item.query_type,
            "features": features_to_array(features),
            "gold_chunks": list(item.gold_chunks),
            "forbidden_chunks": list(item.forbidden_chunks),
            "should_abstain": item.should_abstain,
            "freshness_sensitive": item.freshness_sensitive,
            "annotated_route": item.expected_route.value if item.expected_route else None,
            "restricted_fraction": features.restricted_fraction,
            "routes": {},
        }

        gold = item.gold_set
        for route_name, engine in engines.items():
            route_started = time.perf_counter()
            try:
                result = engine.answer(item.query, item.user_id, write_audit=False)
            except Exception as exc:  # pragma: no cover - defensive
                record["routes"][route_name] = {"error": str(exc)}
                continue
            elapsed_ms = (time.perf_counter() - route_started) * 1000.0
            retrieved = [e.chunk_id for e in result.evidence]

            record["routes"][route_name] = {
                "recall_at_5": recall_at_k(retrieved, gold, 5) if gold else None,
                "ndcg_at_10": ndcg_at_k(retrieved, gold, 10) if gold else None,
                "sufficient": bool(result.sufficiency.sufficient),
                "abstained": bool(result.abstained),
                "evidence": len(result.evidence),
                "acl_violation": bool(set(retrieved) & set(item.forbidden_chunks)),
                "latency_ms": round(elapsed_ms, 3),
            }

        records.append(record)
        if verbose and index % 50 == 0:
            rate = index / (time.perf_counter() - started)
            print(f"    {index}/{len(items)} queries ({rate:.1f}/s)", flush=True)

    if verbose:
        total = time.perf_counter() - started
        print(f"    done: {len(records)} queries x {len(engines)} routes in {total:.1f}s")
    return records


# ===========================================================================
# STEP 2 — derive gold labels from measured outcomes
# ===========================================================================


def assign_gold_labels(records: list[dict]) -> tuple[list[dict], dict]:
    """Label each query with the cheapest route that achieves the best outcome.

    Answerable queries: among routes reaching the maximum recall any route
    achieved, and whose evidence gate passed, take the cheapest. If no route
    passes the gate, take the cheapest that reaches maximum recall anyway — the
    retrieval decision is still informative even when generation would abstain.

    Abstention queries: R0 if no route produced sufficient evidence, which is
    the outcome the label is asserting. If some route *did* find sufficient
    evidence then the annotation and the corpus disagree, and the item is
    dropped rather than used to teach the model something false.

    Queries no route can retrieve any gold chunk for are dropped: they carry no
    signal about which route is preferable.
    """
    labelled: list[dict] = []
    stats = Counter()

    for record in records:
        routes = record["routes"]
        answering = {
            name: info
            for name, info in routes.items()
            if name != "R0" and "error" not in info
        }
        if not answering:
            stats["dropped_all_routes_failed"] += 1
            continue

        if record["should_abstain"]:
            any_sufficient = any(info["sufficient"] for info in answering.values())
            if any_sufficient:
                stats["dropped_abstain_but_answerable"] += 1
                continue
            record["gold_route"] = "R0"
            record["label_basis"] = "no route produced sufficient evidence"
            labelled.append(record)
            stats["labelled_R0"] += 1
            continue

        recalls = {
            name: (info.get("recall_at_5") or 0.0) for name, info in answering.items()
        }
        best_recall = max(recalls.values())
        if best_recall <= 0.0:
            stats["dropped_no_route_retrieves_gold"] += 1
            continue

        winners = [name for name, value in recalls.items() if value >= best_recall]
        gated = [name for name in winners if answering[name]["sufficient"]]
        pool = gated or winners
        chosen = min(pool, key=lambda name: ROUTE_COST[name])

        record["gold_route"] = chosen
        record["label_basis"] = (
            f"cheapest of {sorted(pool)} at recall@5={best_recall:.3f}"
            + ("" if gated else " (no route cleared the evidence gate)")
        )
        record["best_recall"] = best_recall
        labelled.append(record)
        stats[f"labelled_{chosen}"] += 1

    return labelled, dict(stats)


# ===========================================================================
# STEP 3 — splits
# ===========================================================================


def stratified_split(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """60/20/20 split stratified on the gold label.

    The test partition is not used for any threshold, hyperparameter, or model
    selection decision anywhere in this script (§3c).
    """
    rng = np.random.RandomState(SEED)
    by_label: dict[str, list[dict]] = {}
    for record in records:
        by_label.setdefault(record["gold_route"], []).append(record)

    train: list[dict] = []
    validation: list[dict] = []
    test: list[dict] = []
    for label in sorted(by_label):
        group = by_label[label]
        order = rng.permutation(len(group))
        shuffled = [group[i] for i in order]
        n = len(shuffled)
        n_train = int(round(0.6 * n))
        n_val = int(round(0.2 * n))
        # With very few examples of a label, prefer keeping training signal.
        train += shuffled[:n_train]
        validation += shuffled[n_train : n_train + n_val]
        test += shuffled[n_train + n_val :]
    return train, validation, test


def to_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Stack feature vectors and gold labels."""
    features = np.array([r["features"] for r in records], dtype=np.float32)
    labels = np.array([ROUTE_TO_INDEX[r["gold_route"]] for r in records], dtype=np.int32)
    return features, labels


# ===========================================================================
# STEP 4 — train
# ===========================================================================


def train_classifier(x_train, y_train, x_val, y_val, params: dict | None = None):
    """Fit an XGBoost multi-class classifier. Returns ``(model, val_accuracy)``."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  ERROR: xgboost is not installed. pip install xgboost scikit-learn")
        return None, 0.0
    from sklearn.metrics import accuracy_score

    # XGBoost's multi-class objective requires y to contain exactly the
    # contiguous labels 0..k-1, so the observed label set is remapped and
    # inverted on predict. The map is built from the *training* labels only:
    # a class absent from training cannot be predicted, and including it here
    # would make XGBoost reject the fit ("Expected [0 1 2], got [1 2 4]").
    # Validation rows carrying an untrainable label are excluded from the
    # early-stopping eval set but still counted as errors in the accuracy
    # below, so the reported number is not flattered by dropping hard rows.
    observed = sorted(set(y_train.tolist()))
    forward = {label: i for i, label in enumerate(observed)}
    inverse = np.array(observed, dtype=np.int32)
    val_mask = np.array([int(v) in forward for v in y_val], dtype=bool)

    settings = {
        # Single-threaded on purpose. XGBoost's parallel backend spawns loky
        # workers, and on Python 3.14 those die with leaked semaphores partway
        # through the fit, taking the run with them. The training sets here are
        # a few hundred rows, so the threading buys nothing measurable.
        "n_jobs": 1,
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "min_child_weight": 3,
    }
    settings.update(params or {})

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(observed),
        eval_metric="mlogloss",
        random_state=SEED,
        **settings,
    )
    fit_kwargs: dict = {"verbose": False}
    if val_mask.any():
        fit_kwargs["eval_set"] = [
            (
                x_val[val_mask],
                np.array(
                    [forward[int(v)] for v in y_val[val_mask]], dtype=np.int32
                ),
            )
        ]
    model.fit(
        x_train,
        np.array([forward[int(v)] for v in y_train], dtype=np.int32),
        **fit_kwargs,
    )
    model._ahrag_inverse = inverse  # type: ignore[attr-defined]
    predictions = predict_routes(model, x_val)
    return model, float(accuracy_score(y_val, predictions))


def predict_routes(model, x) -> np.ndarray:
    """Predict original route indices, undoing the contiguous label mapping.

    XGBoost's ``predict`` returns class indices for most configurations but a
    probability matrix when ``multi:softprob`` is used with only two observed
    classes, so the shape is normalised before the inverse mapping is applied.
    Indexing a 2-D array through the mapping silently produces a
    multilabel-shaped result that sklearn then rejects several frames later.
    """
    raw = np.asarray(model.predict(x))
    if raw.ndim > 1:
        raw = raw.argmax(axis=1)
    indices = raw.astype(np.int32)
    inverse = getattr(model, "_ahrag_inverse", None)
    if inverse is None:
        return indices
    indices = np.clip(indices, 0, len(inverse) - 1)
    return inverse[indices]


# ===========================================================================
# STEP 5 — baselines that are actually what they claim to be
# ===========================================================================


def rule_based_predictions(base_engine, records: list[dict], router) -> np.ndarray:
    """Ask a real router object for a real decision on each record.

    This is the comparison the previous script claimed to make. It calls
    ``router.decide(features, scope)`` on reconstructed feature objects, so the
    number produced is the router's accuracy rather than the agreement between
    two annotation columns.
    """
    from ahrag.models import AuthorisedScope

    predictions: list[int] = []
    for record in records:
        user = base_engine.get_user(record["user_id"])
        scope: AuthorisedScope = base_engine.acl.scope_for(user)
        probe = base_engine.retrieval.probe(record["query"], scope)
        features = base_engine.features.extract(
            record["query"], user, scope, probe=probe
        )
        decision = router.decide(features, scope)
        predictions.append(ROUTE_TO_INDEX[decision.route.value])
    return np.array(predictions, dtype=np.int32)


def oracle_accuracy(records: list[dict]) -> float:
    """Always 1.0 by construction; reported to anchor the scale."""
    return 1.0


def majority_accuracy(train: list[dict], evaluate: list[dict]) -> float:
    """Accuracy of always predicting the most common training label."""
    if not train or not evaluate:
        return 0.0
    common = Counter(r["gold_route"] for r in train).most_common(1)[0][0]
    return sum(1 for r in evaluate if r["gold_route"] == common) / len(evaluate)


# ===========================================================================
# STEP 5b — the Adaptive-RAG baseline (§6a)
# ===========================================================================


def train_adaptive_rag(train: list[dict], validation: list[dict]):
    """Train the text-only complexity classifier that stands in for Adaptive-RAG.

    Same corpus, same queries, same offline labels as the AHRAG router — the
    only difference is the feature set. That is what makes the P2-versus-B6
    comparison an ablation of the governance features rather than a comparison
    of two unrelated systems.

    Routes are collapsed to Adaptive-RAG's three complexity classes:
    ``A`` (cheapest single-step, R0/R1) → 0, ``B`` (single-step, R2/R3) → 1,
    ``C`` (multi-step, R4) → 2.
    """
    from ahrag.routing.router import TEXT_ONLY_FEATURE_INDICES

    def to_class(route: str) -> int:
        if route in ("R0", "R1"):
            return 0
        if route in ("R2", "R3"):
            return 1
        return 2

    def matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        features = np.array(
            [[r["features"][i] for i in TEXT_ONLY_FEATURE_INDICES] for r in records],
            dtype=np.float32,
        )
        labels = np.array([to_class(r["gold_route"]) for r in records], dtype=np.int32)
        return features, labels

    x_train, y_train = matrix(train)
    x_val, y_val = matrix(validation)
    if len(set(y_train.tolist())) < 2:
        print("  only one complexity class in training data; skipping")
        return None, 0.0
    model, accuracy = train_classifier(x_train, y_train, x_val, y_val)
    return model, accuracy


def adaptive_rag_route_accuracy(model, records: list[dict]) -> float:
    """Route-level accuracy of the Adaptive-RAG baseline on ``records``.

    Scored on routes rather than on its own three classes, so it is directly
    comparable with every other row in the final table.
    """
    if model is None:
        return 0.0
    from ahrag.routing.router import AdaptiveRAGRouter, TEXT_ONLY_FEATURE_INDICES

    features = np.array(
        [[r["features"][i] for i in TEXT_ONLY_FEATURE_INDICES] for r in records],
        dtype=np.float32,
    )
    predicted_classes = predict_routes(model, features)
    correct = 0
    for record, predicted in zip(records, predicted_classes):
        route = AdaptiveRAGRouter.CLASS_TO_ROUTE.get(int(predicted), Route.R3)
        if ROUTE_TO_INDEX[route.value] == ROUTE_TO_INDEX[record["gold_route"]]:
            correct += 1
    return correct / len(records) if records else 0.0


# ===========================================================================
# STEP 6 — lambda sweep (§3d) and learning curve (§3e)
# ===========================================================================


def sweep_lambdas(base_engine, validation: list[dict]) -> list[dict]:
    """Grid-search the utility lambdas on the validation split only.

    The rule-based router's trade-off coefficients (latency 0.09, cost 0.35,
    risk 0.55) were hand-picked. This measures whether they are defensible by
    scoring each candidate against the offline gold labels.
    """
    from ahrag.config import RouterConfig
    from ahrag.routing.router import GovernanceAwareRouter

    y_true = np.array(
        [ROUTE_TO_INDEX[r["gold_route"]] for r in validation], dtype=np.int32
    )
    grid = [
        (latency, cost, risk)
        for latency in (0.03, 0.09, 0.20)
        for cost in (0.12, 0.35, 0.60)
        for risk in (0.25, 0.55, 0.85)
    ]

    results: list[dict] = []
    for latency, cost, risk in grid:
        config = RouterConfig.load(base_engine.settings.router_config)
        config.lambdas.latency = latency
        config.lambdas.cost = cost
        config.lambdas.risk = risk
        router = GovernanceAwareRouter(config, base_engine.settings)
        predictions = rule_based_predictions(base_engine, validation, router)
        accuracy = float((predictions == y_true).mean())

        # Label agreement alone is a poor objective for this router, since it
        # is not what the utility function optimises. So also report what the
        # chosen routes actually *achieved*: the recall each selected route was
        # measured to produce during the offline sweep. No retrieval is re-run
        # -- the per-route outcomes are already in the cached records.
        achieved: list[float] = []
        for record, predicted in zip(validation, predictions):
            route_name = ROUTE_NAMES[int(predicted)]
            info = record["routes"].get(route_name) or {}
            if record["should_abstain"]:
                achieved.append(1.0 if route_name == "R0" else 0.0)
            else:
                achieved.append(float(info.get("recall_at_5") or 0.0))
        results.append(
            {
                "latency": latency,
                "cost": cost,
                "risk": risk,
                "label_agreement": accuracy,
                "achieved_recall_at_5": float(np.mean(achieved)) if achieved else 0.0,
            }
        )
    # Rank by achieved quality, which is the thing worth tuning for.
    results.sort(key=lambda entry: -entry["achieved_recall_at_5"])
    return results


def learning_curve(train: list[dict], validation: list[dict]) -> list[dict]:
    """Validation accuracy as the training set grows (§3e)."""
    x_val, y_val = to_arrays(validation)
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(train))
    shuffled = [train[i] for i in order]

    points: list[dict] = []
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    for fraction in fractions:
        size = max(10, int(len(shuffled) * fraction))
        subset = shuffled[:size]
        if len(set(r["gold_route"] for r in subset)) < 2:
            continue
        x_train, y_train = to_arrays(subset)
        model, accuracy = train_classifier(x_train, y_train, x_val, y_val)
        if model is None:
            break
        points.append({"train_size": size, "val_accuracy": accuracy})
        print(f"    n={size:5d}  validation accuracy={accuracy:.4f}")
    return points


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned router for AHRAG")
    add_corpus_arguments(parser)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run routes and label, but do not train")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of evaluation queries used")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip the lambda grid search (it re-routes the validation split)")
    args = parser.parse_args()

    manifest, eval_set = resolve_corpus(args)

    print("=" * 78)
    print("AHRAG Router Training — offline route labels from measured outcomes")
    print("=" * 78)
    print()

    print("[1/7] Loading corpus and evaluation set...")
    engine = build_seeded_engine(manifest, rebuild_cache=args.rebuild_cache)
    items, _ = load_and_validate(engine, eval_set)
    if args.limit:
        items = items[: args.limit]
        print(f"  limited to {len(items)} queries")
    print(f"  corpus: {len(engine.db.get_documents())} documents, "
          f"{engine.db.count_chunks()} chunks")
    print()

    print("[2/7] Executing all five routes on every query...")
    cache_path = route_records_cache_path(manifest, len(items))
    if cache_path.exists() and not args.rebuild_cache:
        records = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  reusing cached route sweep ({len(records)} queries) from "
              f"{cache_path.name}")
    else:
        records = run_all_routes(engine, items)
        cache_path.write_text(json.dumps(records), encoding="utf-8")
        print(f"  cached route sweep -> {cache_path.name}")
    print()

    print("[3/7] Deriving gold labels from measured outcomes...")
    labelled, label_stats = assign_gold_labels(records)
    print(f"  labelled {len(labelled)} of {len(records)} queries")
    for key in sorted(label_stats):
        print(f"    {key:36s} {label_stats[key]:5d}")
    unanswerable = label_stats.get("dropped_no_route_retrieves_gold", 0)
    contradicted = label_stats.get("dropped_abstain_but_answerable", 0)
    if unanswerable:
        print(f"\n  NOTE: {unanswerable} answerable queries had gold evidence that no")
        print("  route retrieved into the evidence pack. On a corpus this size that is")
        print("  a retrieval-difficulty result, not a labelling bug: it is the")
        print("  selectivity improvement.txt §1 was asking for.")
    if contradicted:
        print(f"\n  NOTE: {contradicted} queries labelled should_abstain had some route")
        print("  produce evidence the sufficiency gate accepted. Dropped from training")
        print("  because the annotation and the corpus disagree, but it is a finding")
        print("  about the gate: at scale there is usually *something* that scores well")
        print("  enough, so the gate abstains less often than it does on the demo corpus.")
    distribution = Counter(r["gold_route"] for r in labelled)
    print(f"  gold route distribution: {dict(sorted(distribution.items()))}")
    if len(distribution) < 2:
        print("\n  Only one route class present. Nothing to learn; stopping.")
        sys.exit(1)
    print()

    print("[4/7] Splitting 60/20/20 (stratified, test held out)...")
    train, validation, test = stratified_split(labelled)
    print(f"  train={len(train)}  validation={len(validation)}  test={len(test)}")
    for name, split in (("train", train), ("val", validation), ("test", test)):
        print(f"    {name:5s}: {dict(sorted(Counter(r['gold_route'] for r in split).items()))}")
    print()

    if args.dry_run:
        print("Dry run complete — labels produced, no training performed.")
        sample = labelled[0]
        print(f"\nSample labelled query: {sample['query'][:70]!r}")
        print(f"  gold_route  : {sample['gold_route']}")
        print(f"  label_basis : {sample['label_basis']}")
        print(f"  per-route recall@5: "
              f"{ {k: v.get('recall_at_5') for k, v in sample['routes'].items()} }")
        return

    print("[5/7] Training the classifier...")
    x_train, y_train = to_arrays(train)
    x_val, y_val = to_arrays(validation)
    model, val_accuracy = train_classifier(x_train, y_train, x_val, y_val)
    if model is None:
        sys.exit(1)
    print(f"  validation accuracy: {val_accuracy:.4f}")

    importances = model.feature_importances_
    ranked = np.argsort(importances)[::-1]
    print("  most informative features:")
    for rank, index in enumerate(ranked[:8], start=1):
        print(f"    {rank}. {FEATURE_NAMES[index]:24s} {importances[index]:.4f}")
    print()

    print("[5b] Training the Adaptive-RAG baseline (text-only features)...")
    adaptive_model, adaptive_val = train_adaptive_rag(train, validation)
    if adaptive_model is not None:
        print(f"  validation complexity-class accuracy: {adaptive_val:.4f}")
    print()

    print("[6/7] Learning curve (validation only)...")
    curve = learning_curve(train, validation)
    print()

    sweep: list[dict] = []
    if not args.skip_sweep:
        print("[6b] Lambda grid search on validation...")
        sweep = sweep_lambdas(engine, validation)
        print("  Ranked by the recall the selected routes actually achieved.")
        print(f"  {'latency':>8s} {'cost':>6s} {'risk':>6s} "
              f"{'achieved R@5':>13s} {'label agree':>12s}")
        for entry in sweep[:5]:
            print(f"  {entry['latency']:8.2f} {entry['cost']:6.2f} "
                  f"{entry['risk']:6.2f} {entry['achieved_recall_at_5']:13.4f} "
                  f"{entry['label_agreement']:12.4f}")
        shipped = next(
            (e for e in sweep
             if (e["latency"], e["cost"], e["risk"]) == (0.09, 0.35, 0.55)),
            None,
        )
        if shipped:
            rank = sweep.index(shipped) + 1
            print(f"  shipped config/router.yaml (0.09/0.35/0.55): "
                  f"achieved R@5={shipped['achieved_recall_at_5']:.4f} "
                  f"-> rank {rank} of {len(sweep)}")
        print()

    print("[7/7] Held-out test evaluation (touched once)...")
    from sklearn.metrics import classification_report

    from ahrag.routing.router import ComplexityOnlyRouter, GovernanceAwareRouter

    x_test, y_test = to_arrays(test)
    ml_predictions = predict_routes(model, x_test)
    ml_accuracy = float((ml_predictions == y_test).mean())

    governance = rule_based_predictions(
        engine, test, GovernanceAwareRouter(engine.config, engine.settings)
    )
    complexity = rule_based_predictions(
        engine, test, ComplexityOnlyRouter(engine.config)
    )

    rows = [
        ("Oracle (upper bound)", oracle_accuracy(test)),
        ("P2  Trained router, all 23 features", ml_accuracy),
        ("P1  GovernanceAwareRouter (shipped, rule-based)", float((governance == y_test).mean())),
        ("B6  Adaptive-RAG (trained, text-only features)",
         adaptive_rag_route_accuracy(adaptive_model, test)),
        ("B5  ComplexityOnlyRouter (hand thresholds)", float((complexity == y_test).mean())),
        ("Majority class", majority_accuracy(train, test)),
    ]
    print()
    print("  READ THIS BEFORE COMPARING THE ROWS BELOW.")
    print("  The label is 'cheapest route that attains the best recall@5 any")
    print("  route attained'. P2 and B6 are *trained to predict that label*.")
    print("  P1 and B5 were not: P1 maximises a utility that trades evidence")
    print("  quality against estimated latency, cost and risk, and B5 routes on")
    print("  query length. Their accuracy here therefore measures disagreement")
    print("  with a labelling rule they never targeted -- it is NOT a statement")
    print("  that they retrieve badly. For retrieval quality compare Recall@5")
    print("  and nDCG@10 in compare_baselines.py, where P1 is competitive.")
    print("  The rows that are directly comparable are P2 vs B6: same corpus,")
    print("  same labels, same model family, differing only in whether the")
    print("  governance and probe features are visible.")
    print()
    print(f"  {'system':48s} {'test accuracy':>14s}")
    print("  " + "-" * 64)
    for name, accuracy in rows:
        print(f"  {name:48s} {accuracy:14.4f}")

    print()
    print("  Route distribution on the held-out test split:")
    print(f"    {'route':6s} {'gold':>6s} {'P2':>6s} {'P1':>6s} {'B5':>6s}")
    for index, name in enumerate(ROUTE_NAMES):
        print(f"    {name:6s} {int((y_test == index).sum()):6d} "
              f"{int((ml_predictions == index).sum()):6d} "
              f"{int((governance == index).sum()):6d} "
              f"{int((complexity == index).sum()):6d}")
    print("  A rule-based router that concentrates on one route while the gold")
    print("  labels concentrate on another scores near zero by construction.")
    print()

    present = sorted(set(y_test.tolist()) | set(ml_predictions.tolist()))
    print()
    print(classification_report(
        y_test, ml_predictions,
        labels=present,
        target_names=[ROUTE_NAMES[i] for i in present],
        zero_division=0,
    ))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(ARTIFACT_DIR / "xgboost_router.json"))
    np.save(ARTIFACT_DIR / "label_mapping.npy", getattr(model, "_ahrag_inverse"))
    if adaptive_model is not None:
        adaptive_model.save_model(str(ARTIFACT_DIR / "adaptive_rag_router.json"))

    metadata = {
        "corpus": str(manifest) if manifest else "packaged seed corpus",
        "eval_set": str(eval_set) if eval_set else "packaged seed suite",
        "corpus_documents": len(engine.db.get_documents()),
        "corpus_chunks": engine.db.count_chunks(),
        "queries_executed": len(records),
        "queries_labelled": len(labelled),
        "label_policy": (
            "Gold route = cheapest route achieving the maximum recall@5 any "
            "route achieved, preferring routes whose evidence gate passed. "
            "Abstention items labelled R0 only when no route produced "
            "sufficient evidence. Unlabelled queries dropped, never defaulted."
        ),
        "label_stats": label_stats,
        "gold_distribution": dict(sorted(distribution.items())),
        "splits": {"train": len(train), "validation": len(validation), "test": len(test)},
        "validation_accuracy": val_accuracy,
        "adaptive_rag_validation_accuracy": adaptive_val,
        "test_accuracy": {name: accuracy for name, accuracy in rows},
        "feature_names": FEATURE_NAMES,
        "route_names": ROUTE_NAMES,
        "feature_importances": {
            FEATURE_NAMES[i]: float(importances[i]) for i in range(len(FEATURE_NAMES))
        },
        "learning_curve": curve,
        "lambda_sweep_top10": sweep[:10],
        "seed": SEED,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Model    -> {ARTIFACT_DIR / 'xgboost_router.json'}")
    print(f"Metadata -> {ARTIFACT_DIR / 'metadata.json'}")
    print()
    print("Enable the trained router with:")
    print("  AHRAG_ROUTER_BACKEND=ml python -m ahrag.evaluate")


if __name__ == "__main__":
    main()
