"""Router feature extraction and the constrained utility function."""

from __future__ import annotations

import pytest

from ahrag.config import RouterConfig
from ahrag.models import AuthorisedScope, Intent, Route, RouterFeatures, User
from ahrag.pipeline import AHRAGEngine
from ahrag.routing.features import FeatureExtractor, ProbeSignals
from ahrag.routing.router import (
    ComplexityOnlyRouter,
    FixedRouter,
    GovernanceAwareRouter,
)


def _scope(allowed: int = 40, total: int = 60, roles: list[str] | None = None) -> AuthorisedScope:
    """Build a synthetic authorised scope for router unit tests."""
    return AuthorisedScope(
        user_id="test.user",
        roles=roles or ["employee"],
        allowed_chunk_ids=[f"c{i}" for i in range(allowed)],
        allowed_doc_ids=["d0"],
        total_chunks=total,
        withheld_count=total - allowed,
        withheld_doc_types=["finance"],
    )


class TestFeatureExtraction:
    """The interpretable feature extractor."""

    @pytest.fixture
    def extractor(self, config: RouterConfig) -> FeatureExtractor:
        return FeatureExtractor(config)

    @pytest.fixture
    def user(self) -> User:
        return User(user_id="u", display_name="U", roles=["employee", "engineering"])

    def test_detects_error_code_identifier(
        self, extractor: FeatureExtractor, user: User
    ) -> None:
        """An error code is recognised and raises the identifier signal."""
        features = extractor.extract(
            "What is the remediation for ERR-5041?", user, _scope()
        )
        assert "ERR-5041" in features.identifiers_found
        assert features.identifier_signal >= 0.6
        assert features.lexical_specificity >= 0.6

    def test_identifier_damps_semantic_ambiguity(
        self, extractor: FeatureExtractor, user: User
    ) -> None:
        """An exact identifier makes a short factual query read as unambiguous.

        Regression guard: interrogative scaffolding ("What is the ... for ...?")
        produces a high stopword ratio, which without damping made identifier
        queries look ambiguous and pushed them away from the sparse route.
        """
        with_id = extractor.extract("What is the remediation for ERR-5041?", user, _scope())
        without_id = extractor.extract("What is the remediation for that?", user, _scope())
        assert with_id.semantic_ambiguity < without_id.semantic_ambiguity

    def test_bare_year_is_not_an_identifier(
        self, extractor: FeatureExtractor, user: User
    ) -> None:
        """A calendar year must not count as an enterprise identifier."""
        features = extractor.extract("What happened in 2026?", user, _scope())
        assert features.identifiers_found == []

    def test_comparison_query_sets_comparison_intent(
        self, extractor: FeatureExtractor, user: User
    ) -> None:
        """Comparison cues produce a comparison intent and multiple hops."""
        features = extractor.extract(
            "Compare the leave policy versus the 2023 edition", user, _scope()
        )
        assert features.intent is Intent.COMPARISON
        assert features.comparison_signal >= 0.5
        assert features.likely_hop_count >= 2

    def test_temporal_query_requires_freshness(
        self, extractor: FeatureExtractor, user: User
    ) -> None:
        """'current ... now' marks the query freshness-sensitive."""
        features = extractor.extract(
            "What is the current leave entitlement now?", user, _scope()
        )
        assert features.temporal_signal > 0
        assert features.freshness_required

    def test_followup_signal_needs_history(
        self, extractor: FeatureExtractor, user: User
    ) -> None:
        """A pronoun is only follow-up evidence when there is a prior turn."""
        without = extractor.extract("What about it?", user, _scope(), history=[])
        with_history = extractor.extract(
            "What about it?", user, _scope(), history=["Tell me about the leave policy"]
        )
        assert without.followup_signal == 0.0
        assert with_history.followup_signal > 0.0

    def test_scope_is_reflected_in_features(
        self, extractor: FeatureExtractor, user: User
    ) -> None:
        """The router sees how much of the corpus was withheld."""
        features = extractor.extract("anything", user, _scope(allowed=10, total=100))
        assert features.authorised_chunk_count == 10
        assert features.restricted_fraction == pytest.approx(0.9)

    def test_probe_absence_biases_toward_abstention(
        self, extractor: FeatureExtractor, user: User
    ) -> None:
        """With no probe evidence, unsupported_signal is maximal."""
        features = extractor.extract("anything at all", user, _scope(), probe=None)
        assert features.unsupported_signal == 1.0

    def test_all_signals_are_normalised(
        self, extractor: FeatureExtractor, user: User
    ) -> None:
        """Every normalised feature stays in 0..1 for a pathological query."""
        query = "compare " * 40 + "ERR-5041 POL-HR-014 2024 2025 ? ? ?"
        features = extractor.extract(query, user, _scope(), probe=ProbeSignals(1.0, 1.0, 1.0))
        for name in (
            "query_length",
            "identifier_signal",
            "numeric_signal",
            "temporal_signal",
            "lexical_specificity",
            "semantic_ambiguity",
            "mixed_signal",
            "comparison_signal",
            "hop_signal",
            "followup_signal",
        ):
            assert 0.0 <= getattr(features, name) <= 1.0, name


class TestHardConstraints:
    """Governance constraints applied before any utility comparison."""

    @pytest.fixture
    def router(self, config: RouterConfig, settings) -> GovernanceAwareRouter:
        return GovernanceAwareRouter(config, settings)

    def test_empty_scope_forces_r0(self, router: GovernanceAwareRouter) -> None:
        """A user who can read nothing gets R0, and no other route is admissible."""
        empty = AuthorisedScope(
            user_id="u",
            roles=["nobody"],
            allowed_chunk_ids=[],
            allowed_doc_ids=[],
            total_chunks=60,
            withheld_count=60,
        )
        features = RouterFeatures(sparse_confidence=0.9, dense_confidence=0.9)
        decision = router.decide(features, empty)

        assert decision.route is Route.R0
        for util in decision.utilities:
            if util.route is not Route.R0:
                assert not util.admissible
                assert util.rejection_reason

    def test_high_confidence_cannot_outbid_empty_scope(
        self, router: GovernanceAwareRouter
    ) -> None:
        """Utility cannot reinstate a route excluded on governance grounds.

        This is the structural property the whole design rests on: the
        inadmissible routes carry higher raw utility here, and still lose.
        """
        empty = AuthorisedScope(
            user_id="u",
            roles=[],
            allowed_chunk_ids=[],
            allowed_doc_ids=[],
            total_chunks=60,
            withheld_count=60,
        )
        features = RouterFeatures(
            sparse_confidence=1.0, dense_confidence=1.0, identifier_signal=1.0
        )
        decision = router.decide(features, empty)
        r1 = decision.utility_of(Route.R1)
        assert r1 is not None and not r1.admissible
        assert decision.route is Route.R0

    def test_low_probe_confidence_forces_r0(self, router: GovernanceAwareRouter) -> None:
        """Below the minimum probe threshold, only abstention is admissible."""
        features = RouterFeatures(sparse_confidence=0.01, dense_confidence=0.01)
        decision = router.decide(features, _scope())
        assert decision.route is Route.R0
        assert decision.hard_constraints_applied

    def test_constraints_are_recorded_for_audit(
        self, router: GovernanceAwareRouter
    ) -> None:
        """Applied constraints are surfaced, not applied silently."""
        features = RouterFeatures(sparse_confidence=0.0, dense_confidence=0.0)
        decision = router.decide(features, _scope())
        assert any("probe" in c for c in decision.hard_constraints_applied)


class TestUtilityFunction:
    """The constrained utility function itself."""

    @pytest.fixture
    def router(self, config: RouterConfig, settings) -> GovernanceAwareRouter:
        return GovernanceAwareRouter(config, settings)

    def test_utility_decomposes_exactly(self, router: GovernanceAwareRouter) -> None:
        """U equals quality minus the three penalties, to floating precision."""
        features = RouterFeatures(sparse_confidence=0.6, dense_confidence=0.5)
        decision = router.decide(features, _scope())
        for util in decision.utilities:
            expected = (
                util.expected_evidence_quality
                - util.latency_penalty
                - util.cost_penalty
                - util.risk_penalty
            )
            assert util.utility == pytest.approx(expected, abs=1e-3)

    def test_every_route_is_scored(self, router: GovernanceAwareRouter) -> None:
        """All five routes appear in the table, including rejected ones."""
        decision = router.decide(
            RouterFeatures(sparse_confidence=0.5, dense_confidence=0.5), _scope()
        )
        assert {u.route for u in decision.utilities} == set(Route)

    def test_identifier_query_prefers_sparse(self, router: GovernanceAwareRouter) -> None:
        """R1 outranks R2 when the query is an exact identifier lookup."""
        features = RouterFeatures(
            identifier_signal=1.0,
            lexical_specificity=0.9,
            semantic_ambiguity=0.1,
            intent=Intent.LOOKUP,
            sparse_confidence=0.85,
            dense_confidence=0.30,
        )
        decision = router.decide(features, _scope())
        r1 = decision.utility_of(Route.R1)
        r2 = decision.utility_of(Route.R2)
        assert r1 and r2 and r1.utility > r2.utility

    def test_conceptual_query_prefers_dense_over_sparse(
        self, router: GovernanceAwareRouter
    ) -> None:
        """R2 outranks R1 when the query is conceptual with no identifier."""
        features = RouterFeatures(
            identifier_signal=0.0,
            lexical_specificity=0.1,
            semantic_ambiguity=0.9,
            intent=Intent.EXPLANATION,
            sparse_confidence=0.25,
            dense_confidence=0.85,
        )
        decision = router.decide(features, _scope())
        r1 = decision.utility_of(Route.R1)
        r2 = decision.utility_of(Route.R2)
        assert r1 and r2 and r2.utility > r1.utility

    def test_multi_hop_query_prefers_r4(self, router: GovernanceAwareRouter) -> None:
        """R4 outranks single-pass routes on a comparison question."""
        features = RouterFeatures(
            hop_signal=1.0,
            comparison_signal=1.0,
            likely_hop_count=3,
            intent=Intent.COMPARISON,
            sparse_confidence=0.6,
            dense_confidence=0.8,
        )
        decision = router.decide(features, _scope())
        r4 = decision.utility_of(Route.R4)
        r1 = decision.utility_of(Route.R1)
        assert r4 and r1 and r4.utility > r1.utility

    def test_restricted_scope_raises_risk_of_answering_routes(
        self, router: GovernanceAwareRouter
    ) -> None:
        """A heavily filtered corpus raises risk for answering routes only.

        ``restricted_fraction`` reaches the router through the feature vector
        (the routing representation ``x``), which the extractor derives from the
        scope, so both are set explicitly here.
        """
        base = RouterFeatures(sparse_confidence=0.7, dense_confidence=0.7)
        open_scope = router.decide(
            base.model_copy(update={"restricted_fraction": 0.0}),
            _scope(allowed=60, total=60),
        )
        tight_scope = router.decide(
            base.model_copy(update={"restricted_fraction": 0.9}),
            _scope(allowed=5, total=60),
        )

        open_r3 = open_scope.utility_of(Route.R3)
        tight_r3 = tight_scope.utility_of(Route.R3)
        assert open_r3 and tight_r3
        assert tight_r3.estimated_risk > open_r3.estimated_risk

        open_r0 = open_scope.utility_of(Route.R0)
        tight_r0 = tight_scope.utility_of(Route.R0)
        assert open_r0 and tight_r0
        assert tight_r0.estimated_risk == pytest.approx(open_r0.estimated_risk)

    def test_freshness_penalises_single_family_routes(
        self, router: GovernanceAwareRouter
    ) -> None:
        """R1/R2 take a larger freshness risk hit than R3/R4."""
        base = RouterFeatures(sparse_confidence=0.7, dense_confidence=0.7)
        fresh = base.model_copy(update={"freshness_required": True})
        stale_decision = router.decide(base, _scope())
        fresh_decision = router.decide(fresh, _scope())

        delta_r1 = (
            fresh_decision.utility_of(Route.R1).estimated_risk
            - stale_decision.utility_of(Route.R1).estimated_risk
        )
        delta_r4 = (
            fresh_decision.utility_of(Route.R4).estimated_risk
            - stale_decision.utility_of(Route.R4).estimated_risk
        )
        assert delta_r1 > delta_r4

    def test_latency_lambda_changes_the_choice(
        self, config: RouterConfig, settings
    ) -> None:
        """Raising λ_latency shifts the choice toward cheaper routes.

        Verifies the operator-tunable Pareto position is a real control and not
        a decorative config value.
        """
        features = RouterFeatures(
            hop_signal=0.8,
            comparison_signal=0.8,
            likely_hop_count=3,
            sparse_confidence=0.7,
            dense_confidence=0.8,
        )
        cheap = GovernanceAwareRouter(config, settings).decide(features, _scope())

        expensive_config = config.model_copy(deep=True)
        expensive_config.lambdas.latency = 5.0
        expensive = GovernanceAwareRouter(expensive_config, settings).decide(
            features, _scope()
        )

        assert cheap.route is Route.R4
        assert expensive.route is not Route.R4

    def test_reasons_are_human_readable(self, router: GovernanceAwareRouter) -> None:
        """Every decision carries a plain-English justification."""
        decision = router.decide(
            RouterFeatures(
                identifier_signal=1.0, sparse_confidence=0.9, dense_confidence=0.3
            ),
            _scope(),
        )
        assert decision.reasons
        assert any("Authorised scope" in r for r in decision.reasons)
        assert any("Utility" in r for r in decision.reasons)

    def test_confidence_reflects_margin(self, router: GovernanceAwareRouter) -> None:
        """A clear winner is more confident than a near tie."""
        clear = router.decide(
            RouterFeatures(
                identifier_signal=1.0,
                lexical_specificity=1.0,
                sparse_confidence=1.0,
                dense_confidence=0.05,
            ),
            _scope(),
        )
        murky = router.decide(
            RouterFeatures(sparse_confidence=0.5, dense_confidence=0.5), _scope()
        )
        assert 0.0 <= clear.confidence <= 1.0
        assert 0.0 <= murky.confidence <= 1.0
        assert clear.confidence > murky.confidence


class TestBaselineRouters:
    """The ablation and fixed routers used by the evaluation harness."""

    def test_fixed_router_always_returns_its_route(self) -> None:
        """A fixed router ignores the features entirely."""
        router = FixedRouter(Route.R1)
        for features in (
            RouterFeatures(comparison_signal=1.0, hop_signal=1.0),
            RouterFeatures(identifier_signal=1.0),
        ):
            assert router.decide(features, _scope()).route is Route.R1

    def test_complexity_router_ignores_governance(self, config: RouterConfig) -> None:
        """The ablation makes the same choice regardless of authorised scope."""
        router = ComplexityOnlyRouter(config)
        features = RouterFeatures(query_length_tokens=20)
        wide = router.decide(features, _scope(allowed=60, total=60))
        narrow = router.decide(features, _scope(allowed=1, total=60))
        assert wide.route is narrow.route

    def test_complexity_router_escalates_on_hops(self, config: RouterConfig) -> None:
        """Multi-hop cues in the text still route it to R4."""
        router = ComplexityOnlyRouter(config)
        decision = router.decide(RouterFeatures(hop_signal=0.9), _scope())
        assert decision.route is Route.R4

    def test_complexity_router_states_what_it_ignores(
        self, config: RouterConfig
    ) -> None:
        """The ablation documents its own blind spots in its reasons."""
        decision = ComplexityOnlyRouter(config).decide(RouterFeatures(), _scope())
        assert any("does not consider" in r for r in decision.reasons)


class TestConfigDriven:
    """Router behaviour must come from config, not hard-coded constants."""

    def test_router_version_comes_from_config(
        self, engine: AHRAGEngine, config: RouterConfig
    ) -> None:
        """The reported router version is the configured one."""
        result = engine.answer("What is ERR-5041?", "alice.employee", write_audit=False)
        assert result.decision.router_version == config.router_version

    def test_missing_config_file_is_an_error(self, tmp_path) -> None:
        """A missing policy file fails loudly rather than defaulting silently."""
        with pytest.raises(FileNotFoundError):
            RouterConfig.load(tmp_path / "nope.yaml")

    def test_malformed_config_is_an_error(self, tmp_path) -> None:
        """A non-mapping policy file is rejected."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError):
            RouterConfig.load(bad)
