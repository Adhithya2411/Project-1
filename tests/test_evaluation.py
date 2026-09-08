"""Evaluation harness: dataset validity, metric correctness, and the run.

The most important test here is ``test_acl_violation_rate_is_zero``: the brief
requires a zero ACL violation rate, and this asserts it from a real end-to-end
run rather than from inspection.
"""

from __future__ import annotations

import pytest

from ahrag.config import RouterConfig, Settings
from ahrag.db import Database
from ahrag.eval.dataset import load_eval_set, query_types
from ahrag.eval.metrics import (
    abstention_appropriate,
    acl_violation,
    aggregate,
    citation_scores,
    dcg,
    freshness_compliant,
    mrr,
    ndcg_at_k,
    percentile,
    recall_at_k,
)
from ahrag.eval.systems import build_systems
from ahrag.evaluate import EVAL_TODAY, run_item, run_system


class TestDataset:
    """The labelled suite itself."""

    def test_loads_and_validates(self) -> None:
        """The packaged suite parses and passes its consistency checks."""
        items = load_eval_set()
        assert len(items) >= 30

    def test_covers_every_required_stratum(self) -> None:
        """All strata the research draft names are represented."""
        types = set(query_types(load_eval_set()))
        assert {
            "exact_lookup",
            "semantic_lookup",
            "procedural",
            "comparison",
            "multi_hop",
            "temporal",
            "permission_constrained",
            "unanswerable",
        } <= types

    def test_gold_chunks_exist_in_the_corpus(self, engine) -> None:
        """Every gold label points at a chunk that actually exists.

        A stale label silently deflates recall for every system at once, which
        looks like a retrieval regression rather than a labelling error.
        """
        known = {c.chunk_id for c in engine.db.get_chunks()}
        for item in load_eval_set():
            for chunk_id in item.gold_chunks + item.forbidden_chunks:
                assert chunk_id in known, f"{item.id} references unknown {chunk_id}"

    def test_gold_chunks_are_authorised_for_their_user(self, engine) -> None:
        """An answerable item's gold evidence is readable by its user."""
        for item in load_eval_set():
            if not item.gold_chunks:
                continue
            user = engine.get_user(item.user_id)
            authorised = set(engine.acl.scope_for(user).allowed_chunk_ids)
            for chunk_id in item.gold_chunks:
                assert chunk_id in authorised, (
                    f"{item.id}: gold {chunk_id} is not readable by {item.user_id}"
                )

    def test_forbidden_chunks_are_genuinely_forbidden(self, engine) -> None:
        """A forbidden label must actually be outside the user's scope."""
        for item in load_eval_set():
            if not item.forbidden_chunks:
                continue
            user = engine.get_user(item.user_id)
            authorised = set(engine.acl.scope_for(user).allowed_chunk_ids)
            for chunk_id in item.forbidden_chunks:
                assert chunk_id not in authorised, (
                    f"{item.id}: {chunk_id} is labelled forbidden but is readable"
                )

    def test_contradictory_labels_are_rejected(self, tmp_path) -> None:
        """An item labelled abstain-with-gold-evidence is a label bug."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "items:\n"
            "  - id: x\n"
            "    query: q\n"
            "    user_id: u\n"
            "    query_type: t\n"
            "    gold_chunks: [a::c000]\n"
            "    should_abstain: true\n"
        )
        with pytest.raises(ValueError, match="should_abstain"):
            load_eval_set(bad)

    def test_duplicate_ids_are_rejected(self, tmp_path) -> None:
        """Duplicate item IDs would silently overwrite results."""
        bad = tmp_path / "dup.yaml"
        bad.write_text(
            "items:\n"
            "  - {id: x, query: q, user_id: u, query_type: t, gold_chunks: [a::c000]}\n"
            "  - {id: x, query: r, user_id: u, query_type: t, gold_chunks: [a::c001]}\n"
        )
        with pytest.raises(ValueError, match="Duplicate"):
            load_eval_set(bad)


class TestMetrics:
    """Metric definitions."""

    def test_recall_at_k(self) -> None:
        """Recall counts gold chunks inside the top k."""
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5
        assert recall_at_k(["x", "a"], {"a"}, 1) == 0.0

    def test_recall_is_none_without_gold(self) -> None:
        """Abstention items contribute nothing to a retrieval average."""
        assert recall_at_k(["a"], set(), 5) is None

    def test_mrr(self) -> None:
        """MRR is the reciprocal rank of the first gold hit."""
        assert mrr(["a", "b"], {"a"}) == 1.0
        assert mrr(["x", "a"], {"a"}) == 0.5
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_dcg_discounts_by_position(self) -> None:
        """A hit at rank 1 is worth more than the same hit at rank 2."""
        assert dcg([1.0, 0.0]) > dcg([0.0, 1.0])

    def test_ndcg_is_one_for_perfect_ranking(self) -> None:
        """A perfect ranking scores 1.0."""
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)

    def test_ndcg_penalises_a_worse_ranking(self) -> None:
        """Pushing gold down the list lowers nDCG."""
        good = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        bad = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        assert good > bad

    def test_citation_scores(self) -> None:
        """Precision, coverage, and groundedness are computed independently."""
        scores = citation_scores(["a", "x"], {"a", "b"}, ["a", "x", "y"])
        assert scores["citation_precision"] == 0.5
        assert scores["citation_coverage"] == 0.5
        assert scores["citation_groundedness"] == 1.0

    def test_groundedness_detects_an_ungrounded_citation(self) -> None:
        """A citation outside the served pack lowers groundedness below 1."""
        scores = citation_scores(["a", "z"], {"a"}, ["a"])
        assert scores["citation_groundedness"] == 0.5

    def test_acl_violation_detects_forbidden_material(self) -> None:
        """Labelled forbidden chunks are caught."""
        violated, offenders = acl_violation(["secret"], [], {"secret"}, {"ok", "secret"})
        assert violated and offenders == ["secret"]

    def test_acl_violation_detects_unlabelled_leak(self) -> None:
        """Anything outside the user's authorised set is caught too.

        The second test is what catches leaks the labels never anticipated.
        """
        violated, offenders = acl_violation(["surprise"], [], set(), {"ok"})
        assert violated and offenders == ["surprise"]

    def test_acl_violation_clean_case(self) -> None:
        """Authorised evidence produces no violation."""
        violated, offenders = acl_violation(["ok"], ["ok"], set(), {"ok"})
        assert not violated and offenders == []

    def test_abstention_appropriateness_is_symmetric(self) -> None:
        """Over-refusing is scored as wrong, not neutral."""
        assert abstention_appropriate(True, True)
        assert abstention_appropriate(False, False)
        assert not abstention_appropriate(False, True)
        assert not abstention_appropriate(True, False)

    def test_freshness_only_applies_to_sensitive_items(self) -> None:
        """Non-sensitive items and abstentions are excluded."""
        assert freshness_compliant(False, ["a"], {"a"}, False) is None
        assert freshness_compliant(True, ["a"], {"a"}, True) is None

    def test_freshness_fails_on_a_superseded_lead_citation(self) -> None:
        """Leading with a retired source is non-compliant."""
        assert freshness_compliant(True, ["old"], {"old"}, False) is False
        assert freshness_compliant(True, ["new", "old"], {"old"}, False) is True

    def test_percentile(self) -> None:
        """p95 interpolates and p50 is the median."""
        values = list(range(1, 101))
        assert percentile(values, 50) == pytest.approx(50.5)
        assert percentile(values, 95) == pytest.approx(95.05, abs=0.2)
        assert percentile([], 95) == 0.0

    def test_aggregate_reports_denominators(self) -> None:
        """Averaged metrics carry the count of contributing items."""
        rows = [
            {"recall_at_5": 1.0, "abstention_appropriate": True, "latency_s": 0.1, "route": "R1"},
            {"recall_at_5": None, "abstention_appropriate": True, "latency_s": 0.3, "route": "R0"},
        ]
        summary = aggregate(rows)
        assert summary["recall_at_5"] == 1.0
        assert summary["recall_at_5_n"] == 1
        assert summary["n"] == 2
        assert summary["route_distribution"]["R1"] == 1

    def test_aggregate_of_nothing(self) -> None:
        """An empty result set aggregates cleanly."""
        assert aggregate([]) == {"n": 0}


class TestEvaluationRun:
    """A real run of the harness."""

    @pytest.fixture(scope="class")
    def evaluated(self, tmp_path_factory):
        """Run all six systems over the full suite once for this class."""
        from .conftest import REPO_ROOT

        data_dir = tmp_path_factory.mktemp("eval")
        settings = Settings(
            data_dir=data_dir,
            db_path=data_dir / "eval.sqlite3",
            router_config=REPO_ROOT / "config" / "router.yaml",
        )
        config = RouterConfig.load(settings.router_config)
        db = Database(settings.db_path)

        from ahrag.pipeline import AHRAGEngine

        bootstrap = AHRAGEngine(settings=settings, config=config, db=db, today=EVAL_TODAY)
        bootstrap.seed(reset=True)

        systems = build_systems(settings, config, db, today=EVAL_TODAY)
        items = load_eval_set()
        return {s.key: run_system(s, items) for s in systems}

    def test_all_six_systems_run(self, evaluated) -> None:
        """The full comparison set executes."""
        assert set(evaluated) == {"B1", "B2", "B3", "B4", "B5", "P1"}

    def test_no_system_errored(self, evaluated) -> None:
        """No query crashed any system."""
        for key, (rows, _) in evaluated.items():
            errors = [r for r in rows if r.get("error")]
            assert not errors, f"{key} errored on {[r['query_id'] for r in errors]}"

    def test_acl_violation_rate_is_zero(self, evaluated) -> None:
        """The headline governance requirement, measured end to end.

        Zero for *every* system, including the baselines: ACL enforcement is
        upstream of routing, so no routing policy can break it. That is the
        point of the invariant, and this test is what verifies it holds rather
        than merely being claimed.
        """
        for key, (rows, summary) in evaluated.items():
            offenders = [
                (r["query_id"], r.get("acl_offenders")) for r in rows if r["acl_violation"]
            ]
            assert summary["acl_violation_rate"] == 0.0, f"{key} leaked: {offenders}"

    def test_citations_are_always_grounded(self, evaluated) -> None:
        """Every citation every system emitted resolves to its evidence pack."""
        for key, (_, summary) in evaluated.items():
            grounded = summary["citation_groundedness"]
            assert grounded is None or grounded == 1.0, key

    def test_fixed_routers_use_only_their_route(self, evaluated) -> None:
        """The fixed baselines really are fixed."""
        for key, route in (("B1", "R1"), ("B2", "R2"), ("B3", "R3"), ("B4", "R4")):
            distribution = evaluated[key][1]["route_distribution"]
            assert distribution[route] == evaluated[key][1]["n"]

    def test_adaptive_routers_use_multiple_routes(self, evaluated) -> None:
        """The adaptive systems actually vary their route."""
        for key in ("B5", "P1"):
            used = {r for r, n in evaluated[key][1]["route_distribution"].items() if n}
            assert len(used) > 1, f"{key} did not adapt"

    def test_governance_router_uses_r0(self, evaluated) -> None:
        """Only the governance-aware router reaches for abstention as a route.

        The complexity-only ablation has no signal that would ever select R0,
        which is precisely the gap the proposed router closes.
        """
        assert evaluated["P1"][1]["route_distribution"]["R0"] > 0
        assert evaluated["B5"][1]["route_distribution"]["R0"] == 0

    def test_every_metric_is_populated(self, evaluated) -> None:
        """The summary reports every metric the brief requires."""
        required = {
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "ndcg_at_10",
            "citation_precision",
            "citation_coverage",
            "abstention_appropriateness",
            "acl_violation_rate",
            "freshness_compliance",
            "mean_latency_s",
            "p95_latency_s",
            "mean_cost_usd",
            "route_distribution",
        }
        for key, (_, summary) in evaluated.items():
            assert required <= set(summary), f"{key} missing {required - set(summary)}"

    def test_latency_and_cost_are_measured_not_zero(self, evaluated) -> None:
        """Efficiency figures come from real execution."""
        for key, (_, summary) in evaluated.items():
            assert summary["mean_latency_s"] > 0, key
            assert summary["mean_cost_usd"] > 0, key
            assert summary["p95_latency_s"] >= summary["mean_latency_s"] * 0.5


class TestSingleItemRun:
    """The per-item runner."""

    def test_run_item_scores_a_permission_item(self, engine) -> None:
        """A permission-constrained item is scored without leaking."""
        from ahrag.eval.systems import SystemSpec

        item = next(i for i in load_eval_set() if i.id == "acl-001")
        spec = SystemSpec(key="T", name="T", description="T", engine=engine)
        row = run_item(spec, item, engine)
        assert row["acl_violation"] is False
        assert row["recall_at_5"] is None
        assert row["query_type"] == "permission_constrained"
