"""Route selection policies.

Three routers share one interface so the evaluation harness can swap them while
holding every other component fixed:

``GovernanceAwareRouter``
    The proposed policy. Applies hard governance constraints to produce the
    admissible route set, then maximises
    ``U(z|x) = Q(z|x) - λ_L·L(z|x) - λ_C·C(z|x) - λ_R·R(z|x)`` over it.

``ComplexityOnlyRouter``
    Ablation standing in for the Adaptive-RAG premise: query-text complexity
    alone picks the route. It gets the identical retrieval stack, reranker,
    evidence gates, and generator — the *only* difference is the routing input.
    That is what makes the comparison an ablation rather than a strawman.

``FixedRouter``
    Always returns one route. Supplies the B1–B4 fixed-pipeline baselines.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ..config import RouterConfig, Settings
from ..models import (
    AuthorisedScope,
    Intent,
    Route,
    RouteDecision,
    RouterFeatures,
    RouteUtility,
)

logger = logging.getLogger(__name__)

_ALL_ROUTES = [Route.R0, Route.R1, Route.R2, Route.R3, Route.R4]


@runtime_checkable
class Router(Protocol):
    """Interface implemented by every routing policy."""

    name: str

    def decide(self, features: RouterFeatures, scope: AuthorisedScope) -> RouteDecision:
        """Choose a route and explain the choice."""


class GovernanceAwareRouter:
    """The proposed governance-aware, cost-aware routing policy.

    Order of operations is the substance of the contribution:

    1. **Hard constraints first.** Authorised scope and minimum probe evidence
       determine which routes are *admissible*. An inadmissible route is
       excluded from the argmax entirely — its utility is never compared, so no
       latency or cost saving can bring it back.
    2. **Utility second.** Among admissible routes only, maximise the
       constrained utility.

    Because step 1 cannot be outbid by step 2, "cheap but unauthorised" is not a
    representable outcome. That is a structural property, not a tuning outcome,
    and it is what the ACL-violation-rate metric in the evaluation checks.
    """

    def __init__(self, config: RouterConfig, settings: Settings) -> None:
        """Bind the routing policy and cost constants."""
        self.config = config
        self.settings = settings
        self.name = "ahrag-governance-aware"

    def decide(self, features: RouterFeatures, scope: AuthorisedScope) -> RouteDecision:
        """Return the chosen route with its full utility table and reasons."""
        constraints_applied: list[str] = []
        admissible = self._admissible_routes(features, scope, constraints_applied)

        utilities = [
            self._score_route(route, features, admissible)
            for route in _ALL_ROUTES
        ]
        candidates = [u for u in utilities if u.admissible]

        if not candidates:
            # Defensive: R0 is always admissible by construction below.
            chosen = next(u for u in utilities if u.route is Route.R0)
            constraints_applied.append("no admissible route; forced R0")
        else:
            chosen = max(candidates, key=lambda u: u.utility)

        fallback = self._fallback_for(chosen.route, candidates)
        confidence = self._confidence(chosen, candidates)
        reasons = self._reasons(chosen, features, scope, constraints_applied, candidates)

        return RouteDecision(
            route=chosen.route,
            fallback_route=fallback,
            confidence=confidence,
            reasons=reasons,
            utilities=utilities,
            features=features,
            estimated_latency_s=chosen.estimated_latency_s,
            estimated_cost_usd=chosen.estimated_cost_usd,
            estimated_evidence_risk=chosen.estimated_risk,
            router_version=self.config.router_version,
            hard_constraints_applied=constraints_applied,
        )

    # -- hard constraints --------------------------------------------------

    def _admissible_routes(
        self,
        features: RouterFeatures,
        scope: AuthorisedScope,
        applied: list[str],
    ) -> dict[Route, str | None]:
        """Return each route mapped to its rejection reason, or None if allowed."""
        rules = self.config.constraints
        result: dict[Route, str | None] = {route: None for route in _ALL_ROUTES}

        if rules.force_r0_on_empty_scope and scope.is_empty:
            applied.append(
                "authorised scope is empty for this user; only R0 is admissible"
            )
            for route in _ALL_ROUTES:
                if route is not Route.R0:
                    result[route] = "no authorised chunks in scope"
            return result

        best_probe = max(features.sparse_confidence, features.dense_confidence)
        if best_probe < rules.min_probe_for_answering:
            applied.append(
                f"best ACL-scoped probe confidence {best_probe:.3f} is below the "
                f"minimum {rules.min_probe_for_answering:.2f}; only R0 is admissible"
            )
            for route in _ALL_ROUTES:
                if route is not Route.R0:
                    result[route] = (
                        f"probe confidence {best_probe:.3f} < "
                        f"{rules.min_probe_for_answering:.2f}"
                    )
        return result

    # -- utility -----------------------------------------------------------

    def _score_route(
        self,
        route: Route,
        features: RouterFeatures,
        admissible: dict[Route, str | None],
    ) -> RouteUtility:
        """Compute the full utility breakdown for one route."""
        rejection = admissible.get(route)
        quality = self._expected_quality(route, features)
        latency = self._estimated_latency(route, features)
        cost = self._estimated_cost(route)
        risk = self._estimated_risk(route, features)

        lambdas = self.config.lambdas
        latency_penalty = lambdas.latency * latency
        cost_penalty = lambdas.cost * (cost * 100.0)  # utility is priced in cents
        risk_penalty = lambdas.risk * risk

        return RouteUtility(
            route=route,
            admissible=rejection is None,
            rejection_reason=rejection,
            expected_evidence_quality=round(quality, 4),
            estimated_latency_s=round(latency, 4),
            estimated_cost_usd=round(cost, 6),
            estimated_risk=round(risk, 4),
            latency_penalty=round(latency_penalty, 4),
            cost_penalty=round(cost_penalty, 4),
            risk_penalty=round(risk_penalty, 4),
            utility=round(quality - latency_penalty - cost_penalty - risk_penalty, 4),
        )

    def _expected_quality(self, route: Route, features: RouterFeatures) -> float:
        """Evaluate the interpretable linear quality model for ``route``."""
        spec = self.config.quality_for(route.value)
        value = spec.prior
        feature_values = self._feature_map(features)
        for name, weight in spec.weights.items():
            value += weight * feature_values.get(name, 0.0)
        value += spec.probe.get("sparse_confidence", 0.0) * features.sparse_confidence
        value += spec.probe.get("dense_confidence", 0.0) * features.dense_confidence
        return max(0.0, min(1.0, value))

    @staticmethod
    def _feature_map(features: RouterFeatures) -> dict[str, float]:
        """Flatten :class:`RouterFeatures` into the names used by the YAML weights."""
        return {
            "query_length": features.query_length,
            "identifier_signal": features.identifier_signal,
            "numeric_signal": features.numeric_signal,
            "temporal_signal": features.temporal_signal,
            "lexical_specificity": features.lexical_specificity,
            "semantic_ambiguity": features.semantic_ambiguity,
            "mixed_signal": features.mixed_signal,
            "comparison_signal": features.comparison_signal,
            "hop_signal": features.hop_signal,
            "followup_signal": features.followup_signal,
            "unsupported_signal": features.unsupported_signal,
            "conflict_likelihood": features.conflict_likelihood,
            "restricted_fraction": features.restricted_fraction,
            "intent_lookup": 1.0 if features.intent is Intent.LOOKUP else 0.0,
            "intent_explanation": 1.0 if features.intent is Intent.EXPLANATION else 0.0,
            "intent_comparison": 1.0 if features.intent is Intent.COMPARISON else 0.0,
            "intent_summarisation": (
                1.0 if features.intent is Intent.SUMMARISATION else 0.0
            ),
            "intent_procedural": 1.0 if features.intent is Intent.PROCEDURAL else 0.0,
            "intent_temporal": 1.0 if features.intent is Intent.TEMPORAL else 0.0,
        }

    def _estimated_latency(self, route: Route, features: RouterFeatures) -> float:
        """Estimate end-to-end latency in seconds for ``route``.

        R4 multiplies by its expected iteration count, which is where the
        quality/efficiency tension the research question asks about actually
        shows up.
        """
        prior = self.config.cost_for(route.value)
        candidates = self.config.retrieval.candidate_top_k
        latency = prior.base_latency_s + prior.per_candidate_latency_s * candidates
        if route is Route.R4:
            iterations = min(
                self.config.retrieval.r4_max_iterations, features.likely_hop_count
            )
            latency *= max(1, iterations)
        return latency

    def _estimated_cost(self, route: Route) -> float:
        """Estimate generation cost in USD for ``route``."""
        prior = self.config.cost_for(route.value)
        return self.settings.estimate_cost_usd(prior.input_tokens, prior.output_tokens)

    def _estimated_risk(self, route: Route, features: RouterFeatures) -> float:
        """Estimate evidence and governance risk for ``route``, in ``0..1``.

        Four additive components: a per-route base rate, a freshness penalty for
        routes that cannot compare versions, a restricted-scope term (a heavily
        filtered view may be a partial picture of the topic), and a low-probe
        term. R0's risk is left at its base rate throughout: abstaining is not
        made riskier by a thin corpus, which is what allows the router to prefer
        it when evidence is genuinely absent.
        """
        model = self.config.risk_model
        risk = model.base.get(route.value, 0.2)

        if features.freshness_required:
            risk += model.freshness_penalty.get(route.value, 0.0)
        if features.conflict_likelihood > 0.0:
            risk += model.conflict_penalty.get(route.value, 0.0) * features.conflict_likelihood

        if route is not Route.R0:
            risk += model.restricted_scope_weight * features.restricted_fraction
            # Evidence risk rises as the best authorised probe hit falls below
            # the reference point. This is the term that lets the router prefer
            # abstention when the *authorised* corpus looks thin, as distinct
            # from when the question is merely difficult.
            reference = max(1e-6, model.low_probe_reference)
            best_probe = max(features.sparse_confidence, features.dense_confidence)
            if best_probe < reference:
                risk += model.low_probe_weight * (reference - best_probe) / reference

        return max(0.0, min(1.0, risk))

    # -- explanation -------------------------------------------------------

    @staticmethod
    def _fallback_for(chosen: Route, candidates: list[RouteUtility]) -> Route:
        """Return the next-best admissible route, or R0."""
        others = sorted(
            (u for u in candidates if u.route is not chosen),
            key=lambda u: u.utility,
            reverse=True,
        )
        return others[0].route if others else Route.R0

    @staticmethod
    def _confidence(chosen: RouteUtility, candidates: list[RouteUtility]) -> float:
        """Confidence from the utility margin over the runner-up.

        A margin of 0.25 or more reads as full confidence; ties read as 0.5.
        Reported so a low-margin decision can be treated with suspicion rather
        than presented as certain.
        """
        others = [u.utility for u in candidates if u.route is not chosen.route]
        if not others:
            return 1.0
        margin = chosen.utility - max(others)
        return round(min(1.0, max(0.0, 0.5 + margin * 2.0)), 3)

    def _reasons(
        self,
        chosen: RouteUtility,
        features: RouterFeatures,
        scope: AuthorisedScope,
        constraints: list[str],
        candidates: list[RouteUtility],
    ) -> list[str]:
        """Build the plain-English justification shown in 'Why this route?'."""
        reasons: list[str] = []
        reasons.extend(f"Hard constraint: {c}." for c in constraints)

        reasons.append(
            f"Authorised scope: {len(scope.allowed_chunk_ids)} of "
            f"{scope.total_chunks} chunks readable by roles "
            f"{', '.join(scope.roles)} "
            f"({scope.withheld_count} withheld before routing)."
        )

        if chosen.route is Route.R0:
            reasons.append(
                "Chose R0 because no admissible retrieval route had enough "
                "authorised evidence to support a grounded answer. R0 may only "
                "clarify or abstain — it never answers from model memory."
            )
        elif chosen.route is Route.R1:
            ids = ", ".join(features.identifiers_found) or "high lexical specificity"
            reasons.append(
                f"Chose R1 (sparse BM25) because the query carries exact lexical "
                f"targets ({ids}). Dense embeddings smooth over rare identifiers; "
                f"BM25 preserves them."
            )
        elif chosen.route is Route.R2:
            reasons.append(
                f"Chose R2 (dense) because the query is conceptual "
                f"(semantic_ambiguity={features.semantic_ambiguity:.2f}) with no "
                f"exact identifier to anchor a lexical match."
            )
        elif chosen.route is Route.R3:
            reasons.append(
                f"Chose R3 (hybrid RRF) because lexical and semantic evidence are "
                f"both informative (mixed_signal={features.mixed_signal:.2f}); "
                f"fusion is worth its extra cost here."
            )
        else:
            reasons.append(
                f"Chose R4 (decomposed iterative hybrid) because the question needs "
                f"about {features.likely_hop_count} retrieval hops "
                f"(hop_signal={features.hop_signal:.2f}, "
                f"comparison_signal={features.comparison_signal:.2f})."
            )

        reasons.append(
            f"Utility {chosen.utility:.3f} = quality {chosen.expected_evidence_quality:.3f} "
            f"− latency {chosen.latency_penalty:.3f} "
            f"− cost {chosen.cost_penalty:.3f} "
            f"− risk {chosen.risk_penalty:.3f}."
        )

        if features.freshness_required:
            reasons.append(
                "Query is freshness-sensitive, so routes that cannot compare "
                "document versions were penalised on risk and current, "
                "higher-authority sources are preferred during evidence packing."
            )
        if features.conflict_likelihood > 0:
            reasons.append(
                "The probe found more than one version of the same policy family; "
                "any conflict will be disclosed rather than silently resolved."
            )
        if scope.restricted_fraction > 0.2:
            reasons.append(
                f"{scope.restricted_fraction:.0%} of the corpus is outside this "
                f"user's scope, which raises evidence risk for every answering "
                f"route (withheld document types: "
                f"{', '.join(scope.withheld_doc_types) or 'none'})."
            )

        runner_up = sorted(
            (u for u in candidates if u.route is not chosen.route),
            key=lambda u: u.utility,
            reverse=True,
        )
        if runner_up:
            best_other = runner_up[0]
            reasons.append(
                f"Runner-up {best_other.route.value} scored "
                f"{best_other.utility:.3f} (margin {chosen.utility - best_other.utility:+.3f})."
            )
        return reasons


class ComplexityOnlyRouter:
    """Ablation: route purely on query-text complexity.

    Mirrors the Adaptive-RAG premise — short question, one hop, no retrieval
    depth; long question, iterative retrieval — with no access to authorised
    scope, freshness, authority, cost, or probe confidence.

    It is held to the same governance floor as everything else: the ACL
    pre-filter still applies (it is upstream of all routers) and the evidence
    gates still run. What it *cannot* do is factor governance into the route
    choice, which is the difference the evaluation measures.
    """

    def __init__(self, config: RouterConfig) -> None:
        """Bind the complexity thresholds."""
        self.config = config
        self.name = "complexity-only"

    def decide(self, features: RouterFeatures, scope: AuthorisedScope) -> RouteDecision:
        """Choose a route from query length and multi-hop cues alone."""
        thresholds = self.config.complexity_baseline
        tokens = features.query_length_tokens

        if features.hop_signal >= 0.5 or features.comparison_signal >= 0.5:
            route = Route.R4
            why = "multi-hop or comparison cues in the query text"
        elif tokens >= thresholds.complex_min_tokens:
            route = Route.R3
            why = f"query is long ({tokens} tokens)"
        elif tokens <= thresholds.simple_max_tokens:
            route = Route.R2
            why = f"query is short ({tokens} tokens)"
        else:
            route = Route.R3
            why = f"query is of moderate length ({tokens} tokens)"

        utilities = [
            RouteUtility(
                route=candidate,
                admissible=True,
                expected_evidence_quality=1.0 if candidate is route else 0.0,
                utility=1.0 if candidate is route else 0.0,
            )
            for candidate in _ALL_ROUTES
        ]
        return RouteDecision(
            route=route,
            fallback_route=Route.R3,
            confidence=0.5,
            reasons=[
                f"Complexity-only baseline: selected {route.value} because {why}.",
                "This router does not consider authorised scope, freshness, "
                "source authority, retrieval probe confidence, latency, or cost.",
            ],
            utilities=utilities,
            features=features,
            router_version="baseline-complexity-only-1.0",
        )


class FixedRouter:
    """Always selects one route. Used for the fixed-pipeline baselines B1–B4."""

    def __init__(self, route: Route, name: str | None = None) -> None:
        """Bind the route this router always returns."""
        self.route = route
        self.name = name or f"fixed-{route.value}"

    def decide(self, features: RouterFeatures, scope: AuthorisedScope) -> RouteDecision:
        """Return the fixed route, with a single explanatory reason."""
        utilities = [
            RouteUtility(
                route=candidate,
                admissible=candidate is self.route,
                rejection_reason=None if candidate is self.route else "fixed pipeline",
                expected_evidence_quality=1.0 if candidate is self.route else 0.0,
                utility=1.0 if candidate is self.route else 0.0,
            )
            for candidate in _ALL_ROUTES
        ]
        return RouteDecision(
            route=self.route,
            fallback_route=self.route,
            confidence=1.0,
            reasons=[f"Fixed pipeline: always uses {self.route.value}."],
            utilities=utilities,
            features=features,
            router_version=f"baseline-fixed-{self.route.value}-1.0",
        )
