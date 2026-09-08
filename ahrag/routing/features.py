"""Interpretable feature extraction for the router.

Every feature is computed by rules a reviewer can read, with the triggering
lexicons and regexes living in ``config/router.yaml`` rather than in code. This
is a deliberate choice, not a shortcut: the research draft proposes a *learned*
router (gradient-boosted or small-LM classifier) trained on offline route
labels. Building that here would mean either shipping an untrained model or
fabricating labels, so the prototype ships the interpretable version and treats
the learned router as an open hypothesis. See ``RESEARCH_LIMITATIONS.md``.

Feature values are normalised to ``0..1`` so the quality weights in the YAML
config are directly comparable to one another.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..config import FeatureParams, RouterConfig
from ..models import AuthorisedScope, Intent, RouterFeatures, User

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_.']*")
_NUMERIC_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"q[1-4]\s*\d{4}|\d{4})\b",
    re.IGNORECASE,
)

# Very common English words carry no retrieval signal; a query made mostly of
# them is a weak lexical target and a strong candidate for the dense route.
_STOPWORDS = frozenset(
    """a an and are as at be been being but by can could do does did for from
    had has have how i if in into is it its me my of on or our should so than
    that the their them then there these they this those to under was we were
    what when where which who whom why will with would you your about above
    after again all am any because before below between both during each few
    further here him his more most no nor not now once only other out over own
    same some such through too very while""".split()
)


@dataclass
class ProbeSignals:
    """First-pass, ACL-scoped retrieval confidence used by the router.

    The probe is a deliberately cheap top-``probe_top_k`` lookup in both
    families. It costs a little latency on every query, and it is what
    distinguishes this router from a purely text-based one: the router learns
    whether the *authorised* corpus actually contains anything relevant before
    it commits to a route, which is precisely what lets it choose abstention on
    governance grounds rather than on question difficulty.
    """

    sparse_confidence: float = 0.0
    dense_confidence: float = 0.0
    agreement: float = 0.0
    sparse_ids: list[str] = field(default_factory=list)
    dense_ids: list[str] = field(default_factory=list)
    families_seen: list[str] = field(default_factory=list)
    conflict_likelihood: float = 0.0
    elapsed_ms: float = 0.0


class FeatureExtractor:
    """Turns a query, its context, and its governance scope into features."""

    def __init__(self, config: RouterConfig) -> None:
        """Bind to the routing policy that supplies lexicons and patterns."""
        self.config = config
        self.params: FeatureParams = config.features

    def extract(
        self,
        query: str,
        user: User,
        scope: AuthorisedScope,
        *,
        history: list[str] | None = None,
        probe: ProbeSignals | None = None,
    ) -> RouterFeatures:
        """Build the full feature vector for one query.

        Args:
            query: Normalised user query.
            user: The requesting user, whose roles define the scope.
            scope: The pre-computed authorised scope (ACL applied already).
            history: Prior turns in the conversation, most recent last.
            probe: First-pass retrieval signals. When absent, probe-derived
                features stay at zero and ``unsupported_signal`` is 1.0, which
                biases the router toward abstention — the safe default when it
                has no evidence that anything relevant is authorised.

        Returns:
            A populated :class:`RouterFeatures`.
        """
        text = query.strip()
        lowered = text.lower()
        tokens = _WORD_RE.findall(lowered)
        token_count = len(tokens)
        probe = probe or ProbeSignals()

        identifiers = self._find_identifiers(text)
        identifier_signal = min(1.0, 0.65 + 0.35 * (len(identifiers) - 1)) if identifiers else 0.0

        numeric_hits = len(_NUMERIC_RE.findall(lowered)) + len(_DATE_RE.findall(lowered))
        numeric_signal = min(1.0, numeric_hits / 3.0)

        temporal_signal = self._lexicon_signal(lowered, self.params.temporal_terms, 2)
        comparison_signal = self._lexicon_signal(lowered, self.params.comparison_terms, 2)
        multihop_signal = self._lexicon_signal(lowered, self.params.multihop_terms, 3)

        content_tokens = [t for t in tokens if t not in _STOPWORDS]
        lexical_specificity = self._lexical_specificity(content_tokens, identifiers)
        semantic_ambiguity = self._semantic_ambiguity(
            tokens, content_tokens, identifier_signal, probe
        )

        intent_scores = self._intent_scores(lowered)
        intent = self._pick_intent(
            intent_scores, comparison_signal, temporal_signal, identifier_signal
        )

        hop_signal, hop_count = self._hop_estimate(
            lowered, tokens, comparison_signal, multihop_signal
        )
        followup_signal = self._followup_signal(lowered, tokens, history or [])

        best_probe = max(probe.sparse_confidence, probe.dense_confidence)
        mixed_signal = self._mixed_signal(identifier_signal, semantic_ambiguity, probe)

        return RouterFeatures(
            raw_query=text,
            query_length_tokens=token_count,
            query_length=min(1.0, token_count / 25.0),
            identifier_signal=identifier_signal,
            identifiers_found=identifiers,
            numeric_signal=numeric_signal,
            temporal_signal=temporal_signal,
            lexical_specificity=lexical_specificity,
            semantic_ambiguity=semantic_ambiguity,
            mixed_signal=mixed_signal,
            intent=intent,
            intent_scores=intent_scores,
            comparison_signal=comparison_signal,
            hop_signal=hop_signal,
            likely_hop_count=hop_count,
            followup_signal=followup_signal,
            user_role=", ".join(user.roles),
            user_roles=list(user.roles),
            authorised_chunk_count=len(scope.allowed_chunk_ids),
            restricted_fraction=scope.restricted_fraction,
            freshness_required=temporal_signal >= 0.34 or intent is Intent.TEMPORAL,
            doc_type_hints=self._doc_type_hints(lowered),
            conflict_likelihood=probe.conflict_likelihood,
            sparse_confidence=probe.sparse_confidence,
            dense_confidence=probe.dense_confidence,
            probe_agreement=probe.agreement,
            unsupported_signal=1.0 - best_probe,
        )

    # -- individual features ----------------------------------------------

    def _find_identifiers(self, text: str) -> list[str]:
        """Return distinct identifier-like strings, in order of appearance."""
        found: list[str] = []
        for pattern in self.params.compiled_identifier_patterns:
            for match in pattern.finditer(text):
                value = match.group(0).strip()
                if value and value not in found:
                    found.append(value)
        return found

    @staticmethod
    def _lexicon_signal(lowered: str, terms: list[str], saturate_at: int) -> float:
        """Fraction of ``saturate_at`` lexicon hits present, capped at 1.0."""
        if not terms or saturate_at <= 0:
            return 0.0
        hits = 0
        for term in terms:
            term_l = term.lower()
            pattern = (
                rf"\b{re.escape(term_l)}\b" if " " not in term_l else re.escape(term_l)
            )
            if re.search(pattern, lowered):
                hits += 1
        return min(1.0, hits / saturate_at)

    @staticmethod
    def _lexical_specificity(content_tokens: list[str], identifiers: list[str]) -> float:
        """How lexically distinctive the query is.

        Rewards long, rare-looking, digit-bearing, or hyphenated tokens: the
        surface properties of enterprise identifiers, SKUs, and clause numbers.
        """
        if not content_tokens:
            return 0.0
        score = 0.0
        for token in content_tokens:
            token_score = 0.0
            if len(token) >= 9:
                token_score += 0.5
            elif len(token) >= 6:
                token_score += 0.25
            if any(ch.isdigit() for ch in token):
                token_score += 0.5
            if "-" in token or "_" in token or "." in token:
                token_score += 0.4
            score += min(1.0, token_score)
        base = score / len(content_tokens)
        if identifiers:
            base = max(base, 0.7)
        return min(1.0, base)

    @staticmethod
    def _semantic_ambiguity(
        tokens: list[str],
        content_tokens: list[str],
        identifier_signal: float,
        probe: ProbeSignals,
    ) -> float:
        """Proxy for how much the query needs semantic rather than exact matching.

        Three additive cues: a high stopword ratio (vague phrasing), an absence
        of identifiers, and a probe in which dense retrieval outperformed
        sparse. The last cue is the empirical one — it observes, on the actual
        authorised corpus, that lexical matching is not finding the answer.

        The sum is then damped by the identifier signal. An exact identifier
        resolves ambiguity almost by definition: "what is the remediation for
        ERR-5041?" has a high stopword ratio but is not a vague question, and
        without this damping the interrogative scaffolding of any short factual
        query reads as ambiguity and pushes it away from the sparse route.
        """
        if not tokens:
            return 0.5
        stopword_ratio = 1.0 - (len(content_tokens) / len(tokens))
        vagueness = min(1.0, stopword_ratio * 1.4)
        no_identifier = 1.0 - identifier_signal
        dense_advantage = max(0.0, probe.dense_confidence - probe.sparse_confidence)
        raw = 0.40 * vagueness + 0.35 * no_identifier + 0.55 * dense_advantage
        return min(1.0, raw * (1.0 - 0.75 * identifier_signal))

    @staticmethod
    def _mixed_signal(
        identifier_signal: float, semantic_ambiguity: float, probe: ProbeSignals
    ) -> float:
        """How much the query needs both lexical and semantic evidence.

        Peaks when neither family clearly dominates: either both cue types are
        present in the text, or both probes returned comparably good — and
        different — results. That disagreement is exactly the condition under
        which rank fusion pays for itself.
        """
        textual = min(identifier_signal, 1.0 - abs(identifier_signal - semantic_ambiguity))
        both_confident = min(probe.sparse_confidence, probe.dense_confidence)
        disagreement = 1.0 - probe.agreement if both_confident > 0.2 else 0.0
        return min(1.0, 0.5 * textual + 0.5 * both_confident * (0.5 + 0.5 * disagreement))

    def _intent_scores(self, lowered: str) -> dict[str, float]:
        """Score each intent by marker-phrase matches."""
        scores: dict[str, float] = {}
        for intent_name, markers in self.params.intent_markers.items():
            hits = sum(1 for marker in markers if marker.lower() in lowered)
            scores[intent_name] = min(1.0, hits / 2.0) if hits else 0.0
        return scores

    @staticmethod
    def _pick_intent(
        scores: dict[str, float],
        comparison_signal: float,
        temporal_signal: float,
        identifier_signal: float,
    ) -> Intent:
        """Resolve the intent label, with governance-relevant tie-breaks.

        Comparison and temporal intents win ties: both change which route is
        appropriate *and* which evidence gates apply (source diversity,
        current-version enforcement), so a false negative there is more costly
        than a false positive.
        """
        if comparison_signal >= 0.5:
            return Intent.COMPARISON
        if temporal_signal >= 0.67:
            return Intent.TEMPORAL
        if scores:
            best_name, best_score = max(scores.items(), key=lambda kv: kv[1])
            if best_score > 0:
                try:
                    return Intent(best_name)
                except ValueError:  # pragma: no cover - config typo guard
                    pass
        if identifier_signal > 0.5:
            return Intent.LOOKUP
        return Intent.LOOKUP

    @staticmethod
    def _hop_estimate(
        lowered: str,
        tokens: list[str],
        comparison_signal: float,
        multihop_signal: float,
    ) -> tuple[float, int]:
        """Estimate reasoning depth as a ``0..1`` signal and an integer hop count.

        Counts explicit conjunctions and question marks alongside the multi-hop
        lexicon, then folds in comparison (a comparison is at least two hops by
        construction) and query length.
        """
        conjunctions = len(re.findall(r"\b(?:and|also|plus|as well as)\b", lowered))
        questions = lowered.count("?")
        length_term = min(1.0, max(0, len(tokens) - 10) / 18.0)
        raw = (
            0.34 * multihop_signal
            + 0.34 * comparison_signal
            + 0.18 * min(1.0, conjunctions / 2.0)
            + 0.14 * min(1.0, max(0, questions - 1) / 2.0)
            + 0.20 * length_term
        )
        signal = min(1.0, raw)
        hop_count = 1 + int(math.floor(signal * 2.4))
        # A comparison is at least two hops by construction: both sides have to
        # be retrieved. Without this floor, "compare A versus B" -- which has no
        # conjunctions and few tokens -- estimates a single hop, and R4's
        # latency estimate then understates the work it will actually do.
        if comparison_signal >= 0.5:
            hop_count = max(hop_count, 2)
        return signal, min(3, hop_count)

    def _followup_signal(
        self, lowered: str, tokens: list[str], history: list[str]
    ) -> float:
        """Detect unresolved coreference that depends on conversation history.

        Only counts pronouns as follow-up evidence when there *is* history: a
        standalone question containing "it" is not a follow-up, it is just
        English. Without that guard the feature fires on the first turn of every
        conversation.
        """
        if not history:
            return 0.0
        pronoun_hits = sum(
            1
            for term in self.params.followup_terms
            if re.search(rf"\b{re.escape(term.lower())}\b", lowered)
        )
        brevity = 1.0 if len(tokens) <= 6 else 0.4
        return min(1.0, (pronoun_hits / 2.0) * brevity + (0.25 if len(tokens) <= 4 else 0.0))

    def _doc_type_hints(self, lowered: str) -> list[str]:
        """Return document types the query's vocabulary points at."""
        hints: list[str] = []
        for doc_type, terms in self.params.doc_type_hints.items():
            if any(re.search(rf"\b{re.escape(t.lower())}", lowered) for t in terms):
                hints.append(doc_type)
        return hints
