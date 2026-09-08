"""Retrieval: fusion, reranking, decomposition, and per-route execution."""

from __future__ import annotations

import pytest

from ahrag.models import Route
from ahrag.pipeline import AHRAGEngine
from ahrag.retrieval.decompose import decompose_query, iteration_weights
from ahrag.retrieval.fusion import rank_positions, reciprocal_rank_fusion
from ahrag.retrieval.rerank import LexicalReranker, normalise_query


class TestReciprocalRankFusion:
    """RRF combines rankings, not scores."""

    def test_agreement_beats_single_top_rank(self) -> None:
        """A chunk ranked 2nd by both beats one ranked 1st by only one.

        This is the property that makes fusion worth its cost: corroboration
        across independent retrievers outweighs a single retriever's confidence.
        """
        fused = dict(reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=10))
        assert fused["b"] > fused["a"]
        assert fused["b"] > fused["c"]

    def test_scores_are_scale_free(self) -> None:
        """Only ranks matter; the retrievers' score magnitudes are irrelevant."""
        first = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
        second = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
        assert first == second
        assert [cid for cid, _ in first] == ["x", "y", "z"]

    def test_larger_k_flattens_rank_influence(self) -> None:
        """A larger smoothing constant narrows the gap between adjacent ranks."""
        tight = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
        loose = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
        assert (tight["a"] - tight["b"]) > (loose["a"] - loose["b"])

    def test_weights_discount_later_rankings(self) -> None:
        """Weighted fusion lets R4 discount later decomposition iterations."""
        unweighted = dict(reciprocal_rank_fusion([["a"], ["b"]], k=10))
        weighted = dict(reciprocal_rank_fusion([["a"], ["b"]], k=10, weights=[1.0, 0.1]))
        assert unweighted["a"] == pytest.approx(unweighted["b"])
        assert weighted["a"] > weighted["b"]

    def test_output_is_deterministic_on_ties(self) -> None:
        """Ties break on chunk ID so evaluation runs are reproducible."""
        result = reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60)
        assert [cid for cid, _ in result] == sorted(cid for cid, _ in result)

    def test_rejects_mismatched_weights(self) -> None:
        """A weights/rankings length mismatch is an error, not a silent truncation."""
        with pytest.raises(ValueError, match="weights length"):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])

    def test_rejects_negative_k(self) -> None:
        """A negative smoothing constant is rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            reciprocal_rank_fusion([["a"]], k=-1)

    def test_rank_positions_keeps_best_rank(self) -> None:
        """A repeated ID keeps its best (lowest) rank."""
        assert rank_positions(["a", "b", "a"]) == {"a": 1, "b": 2}


class TestDecomposition:
    """Query decomposition for R4."""

    def test_original_query_is_always_first(self) -> None:
        """R4 degrades to plain hybrid retrieval rather than losing the intent."""
        query = "What is the leave entitlement?"
        assert decompose_query(query)[0] == query

    def test_between_x_and_y_splits_into_entities(self) -> None:
        """'between A and B' produces one sub-query per side, carrying the topic."""
        parts = decompose_query(
            "Compare the carry-over rules between the current policy and the 2023 edition"
        )
        assert len(parts) >= 3
        joined = " ".join(parts[1:]).lower()
        assert "current policy" in joined
        assert "2023 edition" in joined

    def test_versus_splits_the_query(self) -> None:
        """An explicit 'versus' connective is a split point."""
        parts = decompose_query("annual leave policy versus remote working policy")
        assert len(parts) >= 2

    def test_respects_the_iteration_cap(self) -> None:
        """Decomposition never exceeds the configured breadth cap."""
        query = "compare a and b; and also c; and then d; as well as e"
        assert len(decompose_query(query, max_subqueries=2)) <= 2

    def test_rejects_zero_subqueries(self) -> None:
        """A cap below one is a programming error."""
        with pytest.raises(ValueError, match="at least 1"):
            decompose_query("anything", max_subqueries=0)

    def test_empty_query_yields_nothing(self) -> None:
        """An empty query produces no sub-queries rather than an empty string."""
        assert decompose_query("   ") == []

    def test_iteration_weights_decay(self) -> None:
        """Later iterations carry less weight than the original query."""
        weights = iteration_weights(3, decay=0.5)
        assert weights == [1.0, 0.5, 0.25]

    def test_iteration_weights_reject_bad_decay(self) -> None:
        """A decay outside (0, 1] is rejected."""
        with pytest.raises(ValueError, match="decay"):
            iteration_weights(3, decay=0.0)


class TestLexicalReranker:
    """The transparent fallback reranker."""

    @pytest.fixture
    def reranker(self, engine: AHRAGEngine) -> LexicalReranker:
        return LexicalReranker(engine.db.get_chunks())

    def test_exact_identifier_outscores_a_passing_mention(
        self, engine: AHRAGEngine, reranker: LexicalReranker
    ) -> None:
        """The runbook section beats a status report that merely mentions the code."""
        chunks = {c.chunk_id: c for c in engine.db.get_chunks()}
        definition = chunks["doc-eng-payments-runbook::c002"]
        mention = chunks["doc-pmo-atlas-q2::c003"]
        query = "What is the remediation for ERR-5041?"
        assert reranker.score(query, definition) > reranker.score(query, mention)

    def test_bare_year_does_not_count_as_identifier_match(
        self, engine: AHRAGEngine, reranker: LexicalReranker
    ) -> None:
        """A shared calendar year must not inflate the phrase component.

        Regression guard: matching '2026' between an off-topic query and an
        off-topic chunk was producing a high-confidence false positive.
        """
        chunk = next(
            c for c in engine.db.get_chunks() if c.chunk_id == "doc-pmo-atlas-q2::c000"
        )
        components = reranker.explain("Q3 2026 revenue forecast gross margin", chunk)
        assert components["phrase"] == 0.0

    def test_scores_are_bounded(self, engine: AHRAGEngine, reranker: LexicalReranker) -> None:
        """Every score stays inside 0..1 so thresholds mean one thing."""
        chunks = engine.db.get_chunks()
        for query in ("ERR-5041", "leave", "", "a b c d e f g h i j k l"):
            for chunk in chunks[:10]:
                assert 0.0 <= reranker.score(query, chunk) <= 1.0

    def test_unrelated_query_scores_zero(
        self, engine: AHRAGEngine, reranker: LexicalReranker
    ) -> None:
        """A query sharing no terms with the corpus scores zero, not noise."""
        chunk = engine.db.get_chunks()[0]
        assert reranker.score("xylophone quokka zeppelin", chunk) == 0.0

    def test_explain_returns_all_components(
        self, engine: AHRAGEngine, reranker: LexicalReranker
    ) -> None:
        """The score decomposes into inspectable parts."""
        chunk = engine.db.get_chunks()[0]
        components = reranker.explain("annual leave entitlement", chunk)
        assert set(components) == {"score", "overlap", "phrase", "heading", "coverage"}


class TestQueryNormalisation:
    """Follow-up handling."""

    def test_standalone_query_is_unchanged(self) -> None:
        """Without history, the query passes through untouched."""
        assert normalise_query("What is the leave policy?") == "What is the leave policy?"

    def test_short_pronoun_followup_gains_context(self) -> None:
        """A short pronoun-led follow-up is prefixed with the previous turn."""
        result = normalise_query("what about that?", ["annual leave carry-over"])
        assert "annual leave carry-over" in result

    def test_long_query_is_not_prefixed(self) -> None:
        """A self-contained question is not polluted with prior context."""
        query = "How many days of annual leave can a permanent employee carry over"
        assert normalise_query(query, ["something else entirely"]) == query

    def test_whitespace_is_collapsed(self) -> None:
        """Ragged whitespace is normalised."""
        assert normalise_query("  a   b \n c ") == "a b c"


class TestRouteExecution:
    """Per-route retrieval against the seeded corpus."""

    @pytest.mark.parametrize("route", [Route.R1, Route.R2, Route.R3, Route.R4])
    def test_every_route_returns_authorised_candidates(
        self, engine: AHRAGEngine, route: Route
    ) -> None:
        """Each route retrieves, and retrieves only authorised material."""
        user = engine.get_user("alice.employee")
        scope = engine.acl.scope_for(user)
        allowed = set(scope.allowed_chunk_ids)
        candidates, trace = engine.retrieval.retrieve(
            route, "What is the remediation for ERR-5041?", user, scope
        )
        assert candidates, f"{route.value} returned nothing"
        assert all(c.chunk_id in allowed for c in candidates)
        assert trace.route is route

    def test_r0_retrieves_nothing(self, engine: AHRAGEngine) -> None:
        """R0 is structurally non-retrieving."""
        user = engine.get_user("alice.employee")
        scope = engine.acl.scope_for(user)
        candidates, _ = engine.retrieval.retrieve(Route.R0, "anything", user, scope)
        assert candidates == []

    def test_r4_records_its_subqueries(self, engine: AHRAGEngine) -> None:
        """The decomposition trace is captured for display."""
        user = engine.get_user("carol.hr")
        scope = engine.acl.scope_for(user)
        _, trace = engine.retrieval.retrieve(
            Route.R4,
            "Compare the carry-over rules between the current policy and the 2023 edition",
            user,
            scope,
        )
        assert len(trace.subqueries) > 1
        assert trace.iterations == len(trace.subqueries)
        assert len(trace.per_iteration) == trace.iterations

    def test_r4_respects_the_iteration_cap(self, engine: AHRAGEngine) -> None:
        """Decomposition depth never exceeds the configured maximum."""
        user = engine.get_user("carol.hr")
        scope = engine.acl.scope_for(user)
        _, trace = engine.retrieval.retrieve(
            Route.R4,
            "compare a and b; and also c; and then d; as well as e and f",
            user,
            scope,
        )
        assert trace.iterations <= engine.config.retrieval.r4_max_iterations

    def test_hybrid_marks_fusion_provenance(self, engine: AHRAGEngine) -> None:
        """R3 candidates record which families ranked them."""
        user = engine.get_user("alice.employee")
        scope = engine.acl.scope_for(user)
        candidates, _ = engine.retrieval.retrieve(
            Route.R3, "annual leave entitlement", user, scope
        )
        assert candidates
        assert any(
            c.sparse_rank is not None or c.dense_rank is not None for c in candidates
        )

    def test_scores_are_absolute_not_normalised(self, engine: AHRAGEngine) -> None:
        """The top score varies with query quality.

        Regression guard: max-normalising retrieval scores forced every query's
        best chunk to 1.0, which made the evidence-sufficiency relevance floor
        unfalsifiable.
        """
        user = engine.get_user("alice.employee")
        scope = engine.acl.scope_for(user)
        good, _ = engine.retrieval.retrieve(
            Route.R3, "What is the remediation for ERR-5041?", user, scope
        )
        weak, _ = engine.retrieval.retrieve(
            Route.R3, "quokka zeppelin xylophone marmalade", user, scope
        )
        best_good = max((c.score for c in good), default=0.0)
        best_weak = max((c.score for c in weak), default=0.0)
        assert best_good > best_weak
        assert best_good < 1.0 or best_weak == 0.0


class TestProbe:
    """The ACL-scoped first-pass probe that informs the router."""

    def test_probe_is_scoped_to_the_user(self, engine: AHRAGEngine) -> None:
        """A user without access gets no probe hits on restricted material."""
        erin = engine.get_user("erin.contractor")
        probe = engine.retrieval.probe(
            "acquirer timeout capture idempotency", engine.acl.scope_for(erin)
        )
        assert all(not cid.startswith("doc-eng-payments") for cid in probe.sparse_ids)
        assert all(not cid.startswith("doc-eng-payments") for cid in probe.dense_ids)

    def test_probe_confidence_tracks_query_quality(self, engine: AHRAGEngine) -> None:
        """A well-matched query probes higher than a nonsense one."""
        alice = engine.get_user("alice.employee")
        scope = engine.acl.scope_for(alice)
        strong = engine.retrieval.probe("ERR-5041 acquirer timeout", scope)
        weak = engine.retrieval.probe("quokka zeppelin xylophone", scope)
        assert strong.sparse_confidence > weak.sparse_confidence

    def test_probe_detects_version_conflict(self, engine: AHRAGEngine) -> None:
        """Two versions of one policy family raise the conflict signal."""
        carol = engine.get_user("carol.hr")
        probe = engine.retrieval.probe(
            "annual leave carry-over entitlement", engine.acl.scope_for(carol)
        )
        assert probe.conflict_likelihood == 1.0

    def test_empty_scope_probes_nothing(self, engine: AHRAGEngine) -> None:
        """No authorised chunks means no probe signal at all."""
        from ahrag.models import AuthorisedScope

        empty = AuthorisedScope(
            user_id="u",
            roles=[],
            allowed_chunk_ids=[],
            allowed_doc_ids=[],
            total_chunks=10,
            withheld_count=10,
        )
        probe = engine.retrieval.probe("anything", empty)
        assert probe.sparse_confidence == 0.0
        assert probe.dense_confidence == 0.0
