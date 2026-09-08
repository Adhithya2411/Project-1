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
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

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


class LearnedRouter:
    """A trained classifier that proposes, under governance constraints that veto.

    Design position
    ---------------
    improvement.txt §3 asks for a *learned* router, on the grounds that a linear
    model with hand-picked coefficients cannot be called adaptive in the machine
    learning sense. The obvious implementation — predict a route and use it — is
    wrong here, because it would discard the property the whole system is built
    around: that authorisation and evidence sufficiency are hard constraints
    evaluated *before* any preference computation
    (``GovernanceAwareRouter._admissible_routes``).

    So the learned model does not replace the constraint layer, it replaces the
    *utility* layer inside it:

    1. Compute the admissible set from the governance predicates. Identical code
       path to the rule-based router, reused rather than reimplemented.
    2. Ask the classifier for a probability distribution over routes.
    3. Take the argmax **restricted to the admissible set**.

    A model that has learned to love R4 therefore still cannot answer for a
    principal with an empty scope, and a model that mispredicts on a
    low-confidence probe still abstains. The learned component can change which
    admissible route is preferred; it cannot expand what is permitted. The same
    ACL-violation metric that checks the rule-based router checks this one, and
    ``tests/test_routing.py`` asserts the constraint precedence directly.

    Falls back to the rule-based router when the model artifact is missing, so a
    deployment without a trained model degrades to the shipped policy rather
    than failing to route.
    """

    #: Feature order the model was trained on. Must match the training script's
    #: ``FEATURE_NAMES``; a mismatch is a silent, hard-to-find scoring bug, so
    #: the length is asserted at load time.
    FEATURE_COUNT = 23

    def __init__(
        self,
        config: RouterConfig,
        settings: Settings,
        model_path: Path | None = None,
    ) -> None:
        """Load the trained model, or fall back to the rule-based policy.

        Args:
            config: Routing policy, used for the hard constraints.
            settings: Operational settings.
            model_path: Trained XGBoost JSON. Defaults to the path the training
                script writes.
        """
        self.config = config
        self.settings = settings
        self._governance = GovernanceAwareRouter(config, settings)
        self._model = None
        self._labels: list[int] = []
        self.name = "ahrag-learned"

        path = Path(model_path) if model_path else _default_model_path()
        self.model_path = path
        if not path.exists():
            logger.warning(
                "LearnedRouter: no model at %s; falling back to the rule-based "
                "router. Train one with "
                "improvement_files/ml_router_training/train_router.py",
                path,
            )
            self.name = "ahrag-learned(fallback:rule-based)"
            return

        try:
            import xgboost as xgb
        except ImportError:
            logger.warning(
                "LearnedRouter: xgboost is not installed; falling back to the "
                "rule-based router."
            )
            self.name = "ahrag-learned(fallback:rule-based)"
            return

        try:
            model = xgb.XGBClassifier()
            model.load_model(str(path))
            self._model = model
            mapping = path.parent / "label_mapping.npy"
            if mapping.exists():
                self._labels = [int(v) for v in np.load(mapping)]
            else:
                # Without the mapping the model's class indices are assumed to
                # be route indices already.
                self._labels = list(range(len(_ALL_ROUTES)))
        except Exception as exc:  # pragma: no cover - corrupt artifact
            logger.warning(
                "LearnedRouter: could not load %s (%s); falling back to the "
                "rule-based router.",
                path,
                exc,
            )
            self._model = None
            self.name = "ahrag-learned(fallback:rule-based)"

    @property
    def loaded(self) -> bool:
        """True when a trained model is in use rather than the fallback."""
        return self._model is not None

    def decide(self, features: RouterFeatures, scope: AuthorisedScope) -> RouteDecision:
        """Choose the most probable route among the admissible ones."""
        if self._model is None:
            return self._governance.decide(features, scope)

        constraints_applied: list[str] = []
        admissible = self._governance._admissible_routes(
            features, scope, constraints_applied
        )

        vector = np.asarray([_features_to_vector(features)], dtype=np.float32)
        if vector.shape[1] != self.FEATURE_COUNT:  # pragma: no cover - guard
            raise ValueError(
                f"LearnedRouter expected {self.FEATURE_COUNT} features, "
                f"got {vector.shape[1]}"
            )
        try:
            probabilities = self._model.predict_proba(vector)[0]
        except Exception as exc:  # pragma: no cover - runtime model failure
            logger.warning("LearnedRouter prediction failed (%s); using rule-based.", exc)
            return self._governance.decide(features, scope)

        # Map model class index -> route, then mask to the admissible set.
        scores: dict[Route, float] = {route: 0.0 for route in _ALL_ROUTES}
        for class_index, probability in enumerate(probabilities):
            if class_index >= len(self._labels):
                continue
            route_index = self._labels[class_index]
            if 0 <= route_index < len(_ALL_ROUTES):
                scores[_ALL_ROUTES[route_index]] = float(probability)

        utilities = [
            RouteUtility(
                route=route,
                admissible=admissible[route] is None,
                rejection_reason=admissible[route],
                expected_evidence_quality=scores[route],
                utility=scores[route],
            )
            for route in _ALL_ROUTES
        ]
        candidates = [u for u in utilities if u.admissible]
        if not candidates:
            chosen = next(u for u in utilities if u.route is Route.R0)
            constraints_applied.append("no admissible route; forced R0")
        else:
            chosen = max(candidates, key=lambda u: u.utility)

        ranked = sorted(candidates, key=lambda u: -u.utility)
        fallback = ranked[1].route if len(ranked) > 1 else chosen.route
        margin = (
            ranked[0].utility - ranked[1].utility if len(ranked) > 1 else ranked[0].utility
            if ranked
            else 0.0
        )

        reasons = [
            f"Learned router selected {chosen.route.value} with predicted "
            f"probability {chosen.utility:.3f}.",
            "The classifier ranks only the routes the governance constraints "
            "left admissible; it cannot reinstate an excluded route.",
        ]
        if constraints_applied:
            reasons.extend(constraints_applied)
        excluded = [u for u in utilities if not u.admissible]
        if excluded:
            reasons.append(
                "Excluded on governance grounds: "
                + ", ".join(f"{u.route.value} ({u.rejection_reason})" for u in excluded)
            )

        return RouteDecision(
            route=chosen.route,
            fallback_route=fallback,
            confidence=round(min(1.0, max(0.0, margin)), 4),
            reasons=reasons,
            utilities=utilities,
            features=features,
            router_version=f"learned-xgboost-1.0 ({self.model_path.name})",
            hard_constraints_applied=constraints_applied,
        )


def _default_model_path() -> Path:
    """Where ``train_router.py`` writes its model."""
    return (
        Path(__file__).resolve().parents[2]
        / "improvement_files"
        / "ml_router_training"
        / "artifacts"
        / "xgboost_router.json"
    )


def _features_to_vector(features: RouterFeatures) -> list[float]:
    """Flatten features in the exact order the training script uses.

    Kept adjacent to :class:`LearnedRouter` deliberately: this ordering is a
    contract shared with ``improvement_files/ml_router_training/train_router.py``
    and the two must be changed together.
    """
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


def build_router(config: RouterConfig, settings: Settings, backend: str | None = None):
    """Construct the configured routing policy.

    Args:
        config: Routing policy.
        settings: Operational settings; ``router_backend`` is the default.
        backend: ``"governance"``, ``"learned"``, or ``"complexity"``. Overrides
            the setting when given.

    Returns:
        A router implementing :class:`Router`.

    Raises:
        ValueError: If the backend name is unrecognised.
    """
    choice = (backend or getattr(settings, "router_backend", "governance")).strip().lower()
    if choice in {"governance", "governance-aware", "default"}:
        return GovernanceAwareRouter(config, settings)
    if choice in {"learned", "ml", "xgboost"}:
        return LearnedRouter(config, settings)
    if choice in {"complexity", "complexity-only"}:
        return ComplexityOnlyRouter(config)
    if choice in {"adaptive-rag", "adaptive_rag", "adaptiverag"}:
        return AdaptiveRAGRouter(config, settings)
    raise ValueError(
        f"Unknown router backend {choice!r}. Expected: governance, learned, "
        f"complexity, adaptive-rag"
    )


#: Indices into the 23-feature vector that are derived from the query text
#: alone. Excluded: ``conflict_likelihood``, ``restricted_fraction``,
#: ``sparse_confidence``, ``dense_confidence``, ``probe_agreement`` — each is a
#: function of the authorised corpus or of a retrieval probe, which is exactly
#: the information Adaptive-RAG does not have.
TEXT_ONLY_FEATURE_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22)


class AdaptiveRAGRouter:
    """Reimplementation of Adaptive-RAG (Jeong et al., NAACL 2024) as a baseline.

    Why this exists separately from :class:`ComplexityOnlyRouter`
    ------------------------------------------------------------
    improvement.txt §6(a) makes a fair objection: B5 is *described* as "the
    Adaptive-RAG premise" but it is a pair of hand-set token-count thresholds,
    not the published system. Adaptive-RAG's actual contribution is a **trained
    complexity classifier** over query text, supervised by silver labels
    derived from which retrieval strategy actually answered the question. A
    hand-thresholded stand-in is a weaker baseline than the real thing, and
    beating it proves less.

    This class is the real shape of that system, adapted to AHRAG's route set:

    * A trained classifier, not thresholds.
    * **Query text features only** — the eighteen entries of
      :data:`TEXT_ONLY_FEATURE_INDICES`. It cannot see authorised scope, probe
      confidence, freshness, or conflict likelihood.
    * Three complexity classes, as in the paper, mapped onto AHRAG routes:
      ``A`` (no retrieval needed) → R1, ``B`` (single-step) → R3,
      ``C`` (multi-step) → R4.

    One deviation is worth stating plainly, because it is not a detail. In
    Adaptive-RAG class ``A`` means "the language model already knows this, do
    not retrieve". AHRAG's generator is extractive and answers *only* from the
    evidence pack, so answering from parametric memory is not representable
    here. Class ``A`` is therefore mapped to the cheapest retrieving route
    rather than to no retrieval at all. This makes the baseline slightly
    *stronger* than a literal port would be, not weaker.

    The governance floor still applies, for the same reason it applies to every
    system in ``eval/systems.py``: the ACL pre-filter is upstream of routing.
    What this router cannot do is *use* governance information to choose. That
    is the contrast the comparison against P2 is designed to isolate — both are
    trained classifiers over the same corpus and labels, and the only
    difference is which features they are allowed to see.
    """

    #: Complexity class -> route, per the mapping documented above.
    CLASS_TO_ROUTE = {0: Route.R1, 1: Route.R3, 2: Route.R4}

    def __init__(
        self,
        config: RouterConfig,
        settings: Settings,
        model_path: Path | None = None,
    ) -> None:
        """Load the trained complexity classifier, or fall back to thresholds.

        Args:
            config: Routing policy, used only for the complexity thresholds in
                the fallback path.
            settings: Operational settings.
            model_path: Trained model. Defaults to the artifact
                ``train_router.py`` writes alongside the AHRAG router.
        """
        self.config = config
        self.settings = settings
        self._model = None
        self._fallback = ComplexityOnlyRouter(config)
        self.name = "adaptive-rag"

        path = Path(model_path) if model_path else _default_adaptive_rag_path()
        self.model_path = path
        if not path.exists():
            logger.warning(
                "AdaptiveRAGRouter: no model at %s; falling back to threshold "
                "complexity routing. Train one with "
                "improvement_files/ml_router_training/train_router.py",
                path,
            )
            self.name = "adaptive-rag(fallback:thresholds)"
            return
        try:
            import xgboost as xgb

            model = xgb.XGBClassifier()
            model.load_model(str(path))
            self._model = model
        except Exception as exc:
            logger.warning(
                "AdaptiveRAGRouter: could not load %s (%s); using thresholds.",
                path,
                exc,
            )
            self.name = "adaptive-rag(fallback:thresholds)"

    @property
    def loaded(self) -> bool:
        """True when the trained classifier is in use."""
        return self._model is not None

    def decide(self, features: RouterFeatures, scope: AuthorisedScope) -> RouteDecision:
        """Predict a complexity class from query text and map it to a route."""
        if self._model is None:
            return self._fallback.decide(features, scope)

        full = _features_to_vector(features)
        vector = np.asarray(
            [[full[i] for i in TEXT_ONLY_FEATURE_INDICES]], dtype=np.float32
        )
        try:
            probabilities = self._model.predict_proba(vector)[0]
        except Exception as exc:  # pragma: no cover - runtime model failure
            logger.warning("AdaptiveRAGRouter prediction failed (%s); thresholds.", exc)
            return self._fallback.decide(features, scope)

        predicted = int(np.argmax(probabilities))
        route = self.CLASS_TO_ROUTE.get(predicted, Route.R3)
        confidence = float(probabilities[predicted])

        utilities = [
            RouteUtility(
                route=candidate,
                admissible=True,
                expected_evidence_quality=confidence if candidate is route else 0.0,
                utility=confidence if candidate is route else 0.0,
            )
            for candidate in _ALL_ROUTES
        ]
        return RouteDecision(
            route=route,
            fallback_route=Route.R3,
            confidence=round(confidence, 4),
            reasons=[
                f"Adaptive-RAG: predicted complexity class "
                f"{'ABC'[predicted]} (p={confidence:.3f}) -> {route.value}.",
                "Trained on query-text features only. This router has no access "
                "to authorised scope, probe confidence, freshness, authority, "
                "or conflict likelihood.",
            ],
            utilities=utilities,
            features=features,
            router_version=f"adaptive-rag-trained-1.0 ({self.model_path.name})",
        )


def _default_adaptive_rag_path() -> Path:
    """Where ``train_router.py`` writes the Adaptive-RAG baseline model."""
    return (
        Path(__file__).resolve().parents[2]
        / "improvement_files"
        / "ml_router_training"
        / "artifacts"
        / "adaptive_rag_router.json"
    )
