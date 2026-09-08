"""The comparable systems.

All of them share the *same* database, chunking, indexes, embedding
backend, reranker, evidence-sufficiency gates, generator, and ACL enforcement.
The only thing that varies is the routing policy. That is what makes the
comparison an ablation of routing rather than a comparison of unrelated
pipelines, and it is why B1–B5 are not strawmen: every baseline gets the full
governance stack.

One consequence worth stating plainly: because ACL enforcement is upstream of
routing, **every** system here should record a zero ACL violation rate. That is
the intended result. The metric exists to verify the invariant holds under all
of these policies, not to make the proposed router look better than the
baselines — and it applies to the learned router P2 exactly as it does to the
hand-tuned P1, because the learned model ranks admissible routes rather than
deciding admissibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..config import RouterConfig, Settings
from ..db import Database
from ..models import Route
from ..pipeline import AHRAGEngine
from ..routing.router import (
    AdaptiveRAGRouter,
    ComplexityOnlyRouter,
    FixedRouter,
    GovernanceAwareRouter,
    LearnedRouter,
)


@dataclass
class SystemSpec:
    """One system under evaluation."""

    key: str
    name: str
    description: str
    engine: AHRAGEngine


def build_systems(
    settings: Settings,
    config: RouterConfig,
    db: Database,
    today: date | None = None,
) -> list[SystemSpec]:
    """Construct all six systems over one shared database.

    Args:
        settings: Shared operational settings.
        config: Shared routing policy.
        db: Shared, already-seeded database.
        today: Fixed reference date, so freshness metrics do not drift as the
            real date advances.

    Returns:
        The systems in reporting order: B1–B6, then P1 and P2.
    """
    def engine(router) -> AHRAGEngine:
        return AHRAGEngine(
            settings=settings, config=config, db=db, router=router, today=today
        )

    return [
        SystemSpec(
            key="B1",
            name="Fixed BM25",
            description="Sparse-only retrieval on every query (route R1 always).",
            engine=engine(FixedRouter(Route.R1, name="fixed-bm25")),
        ),
        SystemSpec(
            key="B2",
            name="Fixed dense",
            description="Dense vector retrieval on every query (route R2 always).",
            engine=engine(FixedRouter(Route.R2, name="fixed-dense")),
        ),
        SystemSpec(
            key="B3",
            name="Fixed hybrid RRF",
            description="Sparse+dense with RRF on every query (route R3 always).",
            engine=engine(FixedRouter(Route.R3, name="fixed-hybrid")),
        ),
        SystemSpec(
            key="B4",
            name="Always-maximal iterative hybrid",
            description=(
                "Decomposed iterative hybrid on every query (route R4 always). "
                "The upper bound on retrieval effort, and the cost/latency "
                "reference point for whether adaptivity is worth anything."
            ),
            engine=engine(FixedRouter(Route.R4, name="always-maximal")),
        ),
        SystemSpec(
            key="B5",
            name="Complexity-only router",
            description=(
                "Routes on query-text complexity alone, with no access to "
                "authorised scope, freshness, authority, cost, or probe "
                "confidence. The direct ablation of the governance signal."
            ),
            engine=engine(ComplexityOnlyRouter(config)),
        ),
        SystemSpec(
            key="B6",
            name="Adaptive-RAG (trained, text-only)",
            description=(
                "A trained complexity classifier over query-text features "
                "only, which is the actual shape of Jeong et al. (2024) rather "
                "than B5's hand-set token thresholds. Shares P2's corpus, "
                "queries and offline labels, so B6-versus-P2 isolates the "
                "value of the governance features specifically. Degrades to B5 "
                "when no model artifact is present."
            ),
            engine=engine(AdaptiveRAGRouter(config, settings)),
        ),
        SystemSpec(
            key="P1",
            name="AHRAG governance-aware router",
            description=(
                "Hard governance constraints first, then constrained utility "
                "maximisation over the admissible routes."
            ),
            engine=engine(GovernanceAwareRouter(config, settings)),
        ),
        SystemSpec(
            key="P2",
            name="AHRAG learned router",
            description=(
                "Same hard governance constraints, but the admissible routes "
                "are ranked by a trained classifier instead of the hand-set "
                "utility weights. Degrades to P1 when no model artifact is "
                "present, so its row is only meaningful after "
                "improvement_files/ml_router_training/train_router.py has run."
            ),
            engine=engine(LearnedRouter(config, settings)),
        ),
    ]
