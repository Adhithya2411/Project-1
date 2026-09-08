"""Configuration for AHRAG.

Two layers:

``Settings``
    Environment / ``.env`` driven operational settings (paths, backends, keys).
    Pydantic-settings; see ``.env.example``.

``RouterConfig``
    The routing *policy*: lambda coefficients, per-route cost priors, quality
    weights, risk model, hard constraints, retrieval parameters, and evidence
    thresholds. Loaded from ``config/router.yaml`` so the policy can be changed
    and re-evaluated without touching Python.

Both are plain objects passed explicitly to the components that need them. No
module-level singletons are created at import time; ``get_settings()`` caches
per-process but can be bypassed by constructing ``Settings`` directly, which is
what the tests do.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Operational settings resolved from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="AHRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = REPO_ROOT / "data"
    db_path: Path = REPO_ROOT / "data" / "ahrag.sqlite3"
    router_config: Path = REPO_ROOT / "config" / "router.yaml"

    embedding_backend: str = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    vector_store: str = "auto"
    reranker: str = "auto"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_target_chars: int = 900
    chunk_overlap_chars: int = 150

    # Routing policy. "governance" is the shipped rule-based router;
    # "learned" uses the trained classifier from
    # improvement_files/ml_router_training/ and degrades to "governance"
    # when no model artifact is present. "complexity" is the Adaptive-RAG
    # style ablation. In every case the hard governance constraints run
    # first; the backend only changes how admissible routes are ranked.
    router_backend: str = "governance"

    # -- Scope-Pure Index Specialisation (see ahrag/index/scoped.py) --------
    # Off by default: the unspecialised system is the reference point for the
    # ablation, and leaving it as the default keeps every existing result
    # reproducible.
    index_specialisation: bool = False
    # Information-flow budget for the *sparse* channel, not an ordinary
    # hyperparameter. 0.0 leaves class-local IDF untouched and is provably
    # non-interfering; 1.0 restores the global IDF values. Anything in between
    # trades purity for estimator stability on small scopes.
    # Note the scope of this knob: it governs BM25 IDF only. The LSA basis is
    # class-local whenever specialisation is on, so lambda=1 does *not*
    # reproduce the unspecialised system. A complete frontier would need a
    # dense analogue; see RESEARCH_LIMITATIONS.md.
    specialisation_lambda: float = 0.0
    # LRU cap on cached per-class indexes. Bounds memory against a corpus whose
    # ACL lattice is wider than expected; excess classes fall back to global.
    specialisation_max_classes: int = 8
    # Per-class recalibration of the BM25 saturation constant is a *separate*
    # change from index purity, and it is off by default for a specific reason:
    # `min_probe_for_answering` in config/router.yaml was hand-tuned against the
    # historic constant 6.0, so rescaling confidence silently invalidates that
    # threshold. Enabling this requires re-tuning the threshold jointly, which
    # is the search improvement.txt §3(d) asks for and which has not been run.
    # Left on with the default threshold, it shifts routes for reasons that have
    # nothing to do with retrieval quality.
    specialisation_calibrate_confidence: bool = False
    # Quantile of a class's own match-score distribution taken as the
    # saturation point. 0.25 mirrors where the historic constant 6.0 sits in the
    # observed distribution of real evaluation-query top-1 scores (22nd
    # percentile on the seed corpus).
    specialisation_confidence_quantile: float = 0.25

    anthropic_model: str = "claude-sonnet-5"
    generation_max_tokens: int = 800
    cost_per_mtok_input: float = 3.0
    cost_per_mtok_output: float = 15.0

    verbose_audit: bool = False
    log_level: str = "INFO"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"

    random_seed: int = 1729

    @property
    def anthropic_api_key(self) -> str | None:
        """Return ``ANTHROPIC_API_KEY`` from the environment, if present.

        Read lazily rather than bound as a field so that tests can toggle it
        with ``monkeypatch.setenv`` without rebuilding ``Settings``.
        """
        import os

        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        return key or None

    def estimate_cost_usd(self, input_tokens: float, output_tokens: float) -> float:
        """Convert a token budget into an estimated USD cost.

        These are local accounting constants for comparing routes against each
        other, not a billing figure. When the extractive generator is used the
        true monetary cost is zero; the estimate is still reported so that route
        comparisons remain meaningful across generator backends.
        """
        return (
            input_tokens * self.cost_per_mtok_input
            + output_tokens * self.cost_per_mtok_output
        ) / 1_000_000.0


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-cached settings. Construct ``Settings()`` to bypass."""
    return Settings()


# ---------------------------------------------------------------------------
# Router policy configuration
# ---------------------------------------------------------------------------


class Lambdas(BaseModel):
    """Trade-off coefficients of the constrained utility function."""

    latency: float = 0.09
    cost: float = 0.35
    risk: float = 0.55


class RouteCost(BaseModel):
    """Operational priors for one route."""

    base_latency_s: float = 0.2
    per_candidate_latency_s: float = 0.003
    input_tokens: int = 1500
    output_tokens: int = 260


class QualitySpec(BaseModel):
    """Interpretable linear model for a route's expected evidence quality."""

    prior: float = 0.3
    weights: dict[str, float] = Field(default_factory=dict)
    probe: dict[str, float] = Field(default_factory=dict)


class RiskModel(BaseModel):
    """Additive risk model, all terms in ``0..1`` before clipping."""

    base: dict[str, float] = Field(default_factory=dict)
    freshness_penalty: dict[str, float] = Field(default_factory=dict)
    restricted_scope_weight: float = 0.30
    low_probe_weight: float = 0.45
    low_probe_reference: float = 0.65
    conflict_penalty: dict[str, float] = Field(default_factory=dict)


class Constraints(BaseModel):
    """Hard governance constraints evaluated *before* any utility comparison."""

    force_r0_on_empty_scope: bool = True
    min_probe_for_answering: float = 0.12
    r0_is_non_generative: bool = True


class RetrievalParams(BaseModel):
    """Retrieval depth and fusion parameters."""

    probe_top_k: int = 5
    candidate_top_k: int = 30
    rrf_k: int = 60
    rerank_top_k: int = 12
    evidence_top_k: int = 6
    r4_max_iterations: int = 3
    r4_subquery_top_k: int = 8


class EvidenceParams(BaseModel):
    """Thresholds for the evidence-sufficiency gate."""

    min_top_score: float = 0.18
    min_supporting_chunks: int = 1
    min_mean_score: float = 0.10
    diversity_required_intents: list[str] = Field(default_factory=list)
    min_distinct_documents: int = 2
    min_authority_for_policy: int = 3
    demote_superseded: bool = True
    superseded_score_multiplier: float = 0.55
    enforce_current_version_on_freshness: bool = True


class FeatureParams(BaseModel):
    """Lexicons and patterns used by the interpretable feature extractor."""

    identifier_patterns: list[str] = Field(default_factory=list)
    temporal_terms: list[str] = Field(default_factory=list)
    comparison_terms: list[str] = Field(default_factory=list)
    multihop_terms: list[str] = Field(default_factory=list)
    followup_terms: list[str] = Field(default_factory=list)
    intent_markers: dict[str, list[str]] = Field(default_factory=dict)
    doc_type_hints: dict[str, list[str]] = Field(default_factory=dict)

    @functools.cached_property
    def compiled_identifier_patterns(self) -> list[re.Pattern[str]]:
        """Identifier regexes compiled once, case-insensitive."""
        compiled: list[re.Pattern[str]] = []
        for pattern in self.identifier_patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:  # pragma: no cover - config error path
                raise ValueError(
                    f"Invalid identifier_pattern in router config: {pattern!r} ({exc})"
                ) from exc
        return compiled

    model_config = {"ignored_types": (functools.cached_property,)}


class ComplexityBaseline(BaseModel):
    """Token thresholds for the complexity-only ablation router (baseline B5)."""

    simple_max_tokens: int = 6
    complex_min_tokens: int = 16


class RouterConfig(BaseModel):
    """The complete routing policy, loaded from YAML."""

    router_version: str = "ahrag-router-dev"
    lambdas: Lambdas = Field(default_factory=Lambdas)
    route_costs: dict[str, RouteCost] = Field(default_factory=dict)
    quality_model: dict[str, QualitySpec] = Field(default_factory=dict)
    risk_model: RiskModel = Field(default_factory=RiskModel)
    constraints: Constraints = Field(default_factory=Constraints)
    retrieval: RetrievalParams = Field(default_factory=RetrievalParams)
    evidence: EvidenceParams = Field(default_factory=EvidenceParams)
    features: FeatureParams = Field(default_factory=FeatureParams)
    complexity_baseline: ComplexityBaseline = Field(default_factory=ComplexityBaseline)

    @classmethod
    def load(cls, path: str | Path | None = None) -> RouterConfig:
        """Load and validate the router policy from ``path``.

        Falls back to the packaged default location when ``path`` is None. A
        missing file is a hard error rather than a silent default, because a
        silently-defaulted routing policy would make evaluation results
        unreproducible.
        """
        target = Path(path) if path is not None else REPO_ROOT / "config" / "router.yaml"
        if not target.exists():
            raise FileNotFoundError(
                f"Router config not found at {target}. Copy config/router.yaml or set "
                "AHRAG_ROUTER_CONFIG."
            )
        try:
            raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Router config {target} is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Router config {target} must contain a YAML mapping.")
        return cls.model_validate(raw)

    def cost_for(self, route_value: str) -> RouteCost:
        """Return the operational prior for a route, defaulting conservatively."""
        return self.route_costs.get(route_value, RouteCost())

    def quality_for(self, route_value: str) -> QualitySpec:
        """Return the quality model for a route, defaulting conservatively."""
        return self.quality_model.get(route_value, QualitySpec())
