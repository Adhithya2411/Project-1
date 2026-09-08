"""Evidence sufficiency, conflict detection, and grounded generation."""

from __future__ import annotations

import pytest

from ahrag.evidence.conflict import ConflictDetector
from ahrag.generation.base import verify_citations
from ahrag.generation.extractive import ExtractiveGenerator
from ahrag.models import AbstentionReason, Intent, RouterFeatures, ScoredChunk
from ahrag.pipeline import AHRAGEngine


def _scored(engine: AHRAGEngine, chunk_ids: list[str], score: float = 0.6) -> list[ScoredChunk]:
    """Build scored chunks from IDs for gate/detector unit tests."""
    by_id = {c.chunk_id: c for c in engine.db.get_chunks()}
    return [ScoredChunk(chunk=by_id[cid], score=score) for cid in chunk_ids]


class TestSufficiencyGate:
    """The gate that stands between retrieval and generation."""

    def test_empty_evidence_is_insufficient(self, engine: AHRAGEngine) -> None:
        """No evidence means no answer, with a specific reason."""
        user = engine.get_user("alice.employee")
        report = engine.validator.validate([], user, RouterFeatures())
        assert not report.sufficient
        assert report.failure_reason is AbstentionReason.NO_AUTHORISED_EVIDENCE

    def test_low_relevance_is_insufficient(self, engine: AHRAGEngine) -> None:
        """A pack below the relevance floor is refused."""
        user = engine.get_user("alice.employee")
        evidence = _scored(engine, ["doc-hr-remote::c002"], score=0.01)
        report = engine.validator.validate(evidence, user, RouterFeatures())
        assert not report.sufficient
        assert report.failure_reason is AbstentionReason.LOW_RELEVANCE

    def test_good_evidence_passes(self, engine: AHRAGEngine) -> None:
        """A relevant, authorised, current pack passes every check."""
        user = engine.get_user("alice.employee")
        evidence = _scored(engine, ["doc-hr-remote::c002"], score=0.8)
        report = engine.validator.validate(evidence, user, RouterFeatures())
        assert report.sufficient
        assert all(report.checks.values())

    def test_unauthorised_chunk_fails_the_gate(self, engine: AHRAGEngine) -> None:
        """The gate re-verifies ACL rather than trusting upstream filtering."""
        alice = engine.get_user("alice.employee")
        evidence = _scored(engine, ["doc-fin-q3-forecast::c001"], score=0.9)
        report = engine.validator.validate(evidence, alice, RouterFeatures())
        assert not report.sufficient
        assert report.failure_reason is AbstentionReason.NO_AUTHORISED_EVIDENCE
        assert not report.checks["acl_validated"]

    def test_comparison_requires_multiple_documents(self, engine: AHRAGEngine) -> None:
        """A comparison grounded in one source is not a comparison."""
        user = engine.get_user("carol.hr")
        features = RouterFeatures(intent=Intent.COMPARISON)
        single = _scored(engine, ["doc-hr-leave-v2::c002"], score=0.8)
        report = engine.validator.validate(single, user, features)
        assert not report.sufficient
        assert report.failure_reason is AbstentionReason.INSUFFICIENT_DIVERSITY

        both = _scored(
            engine, ["doc-hr-leave-v2::c002", "doc-hr-leave-v1::c003"], score=0.8
        )
        assert engine.validator.validate(both, user, features).sufficient

    def test_low_mean_score_is_insufficient(self, engine: AHRAGEngine) -> None:
        """One marginal hit among noise does not ground an answer."""
        user = engine.get_user("alice.employee")
        by_id = {c.chunk_id: c for c in engine.db.get_chunks()}
        evidence = [
            ScoredChunk(chunk=by_id["doc-hr-remote::c002"], score=0.11),
            ScoredChunk(chunk=by_id["doc-hr-remote::c003"], score=0.02),
            ScoredChunk(chunk=by_id["doc-hr-remote::c004"], score=0.02),
        ]
        report = engine.validator.validate(evidence, user, RouterFeatures())
        assert not report.sufficient
        assert report.failure_reason is AbstentionReason.LOW_RELEVANCE

    def test_clarifying_question_does_not_leak(self, engine: AHRAGEngine) -> None:
        """With no authorised evidence, suggestions stay generic."""
        question = engine.validator.clarifying_question(RouterFeatures(), [])
        titles = {meta.title for meta in engine.db.get_documents()}
        for title in titles:
            assert title not in question

    def test_clarifying_question_suggests_authorised_sources(
        self, engine: AHRAGEngine
    ) -> None:
        """When there is authorised evidence, it may be named."""
        evidence = _scored(engine, ["doc-hr-remote::c002"])
        question = engine.validator.clarifying_question(RouterFeatures(), evidence)
        assert "Hybrid and Remote Working Policy" in question


class TestEvidencePacking:
    """Selection of the final evidence pack."""

    def test_zero_scored_chunks_are_dropped(self, engine: AHRAGEngine) -> None:
        """Padding chunks would dilute the mean-score check into meaninglessness."""
        user = engine.get_user("alice.employee")
        by_id = {c.chunk_id: c for c in engine.db.get_chunks()}
        candidates = [
            ScoredChunk(chunk=by_id["doc-hr-remote::c002"], score=0.5),
            ScoredChunk(chunk=by_id["doc-hr-remote::c003"], score=0.0),
        ]
        packed = engine.validator.pack(candidates, user, top_k=5, freshness_sensitive=False)
        assert [p.chunk_id for p in packed] == ["doc-hr-remote::c002"]

    def test_pack_respects_top_k(self, engine: AHRAGEngine) -> None:
        """The pack never exceeds the configured size."""
        user = engine.get_user("carol.hr")
        candidates = _scored(
            engine,
            [c.chunk_id for c in engine.db.get_chunks() if c.doc_id == "doc-hr-leave-v2"],
            score=0.5,
        )
        packed = engine.validator.pack(candidates, user, top_k=2, freshness_sensitive=False)
        assert len(packed) == 2

    def test_pack_filters_unauthorised(self, engine: AHRAGEngine) -> None:
        """Packing is another place ACL is enforced, not assumed."""
        alice = engine.get_user("alice.employee")
        candidates = _scored(
            engine, ["doc-hr-remote::c002", "doc-fin-q3-forecast::c001"], score=0.7
        )
        packed = engine.validator.pack(candidates, alice, top_k=5, freshness_sensitive=False)
        assert all(not p.chunk_id.startswith("doc-fin-q3") for p in packed)

    def test_freshness_reorders_toward_current_version(
        self, engine: AHRAGEngine
    ) -> None:
        """With equal relevance, the live policy outranks the retired one."""
        user = engine.get_user("carol.hr")
        candidates = _scored(
            engine, ["doc-hr-leave-v1::c002", "doc-hr-leave-v2::c001"], score=0.5
        )
        packed = engine.validator.pack(candidates, user, top_k=5, freshness_sensitive=True)
        assert packed[0].chunk.doc_id == "doc-hr-leave-v2"


class TestConflictDetection:
    """Disclosure of disagreement between authorised sources."""

    @pytest.fixture
    def detector(self, engine: AHRAGEngine) -> ConflictDetector:
        return ConflictDetector(engine.freshness)

    def test_version_conflict_is_reported(
        self, engine: AHRAGEngine, detector: ConflictDetector
    ) -> None:
        """Two versions of one family produce a version conflict."""
        evidence = _scored(
            engine, ["doc-hr-leave-v2::c002", "doc-hr-leave-v1::c003"]
        )
        conflicts = detector.detect(evidence)
        assert any(c.kind == "version" for c in conflicts)
        version = next(c for c in conflicts if c.kind == "version")
        assert version.policy_family == "leave-policy"
        assert version.preferred_chunk_id.startswith("doc-hr-leave-v2")

    def test_numeric_conflict_is_reported(
        self, engine: AHRAGEngine, detector: ConflictDetector
    ) -> None:
        """26 days versus 22 days is detected as a numeric disagreement."""
        evidence = _scored(
            engine, ["doc-hr-leave-v2::c001", "doc-hr-leave-v1::c002"]
        )
        numeric = [c for c in detector.detect(evidence) if c.kind == "numeric"]
        assert numeric
        assert any("26" in c.description and "22" in c.description for c in numeric)

    def test_both_sides_stay_in_the_report(
        self, engine: AHRAGEngine, detector: ConflictDetector
    ) -> None:
        """Conflict is disclosed, not resolved: the loser is still named."""
        evidence = _scored(
            engine, ["doc-hr-leave-v2::c002", "doc-hr-leave-v1::c003"]
        )
        conflict = next(c for c in detector.detect(evidence) if c.kind == "version")
        assert len(conflict.chunk_ids) == 2
        assert conflict.preference_reason

    def test_single_version_produces_no_conflict(
        self, engine: AHRAGEngine, detector: ConflictDetector
    ) -> None:
        """One version of a policy is not a conflict."""
        evidence = _scored(engine, ["doc-hr-leave-v2::c002"])
        assert detector.detect(evidence) == []

    def test_unrelated_documents_produce_no_conflict(
        self, engine: AHRAGEngine, detector: ConflictDetector
    ) -> None:
        """Different policy families never conflict with each other."""
        evidence = _scored(engine, ["doc-hr-remote::c002", "doc-fin-expenses::c004"])
        assert detector.detect(evidence) == []


class TestGeneration:
    """Grounded generation and citation verification."""

    def test_every_answer_line_carries_a_citation(self, engine: AHRAGEngine) -> None:
        """The extractive generator cites every statement it emits."""
        evidence = _scored(engine, ["doc-hr-leave-v2::c001"], score=0.9)
        answer = ExtractiveGenerator().generate("annual leave entitlement", evidence)
        for line in answer.text.splitlines():
            if line.strip().startswith("- "):
                assert "[" in line and "]" in line

    def test_citations_resolve_to_served_evidence(self, engine: AHRAGEngine) -> None:
        """Every citation maps to a chunk in the pack."""
        evidence = _scored(
            engine, ["doc-hr-leave-v2::c001", "doc-hr-leave-v2::c002"], score=0.9
        )
        answer = ExtractiveGenerator().generate("annual leave carry over", evidence)
        served = {e.chunk_id for e in evidence}
        assert answer.citations
        assert all(c.chunk_id in served for c in answer.citations)

    def test_output_is_deterministic(self, engine: AHRAGEngine) -> None:
        """The same inputs produce byte-identical output."""
        evidence = _scored(engine, ["doc-hr-leave-v2::c001"], score=0.9)
        generator = ExtractiveGenerator()
        first = generator.generate("annual leave entitlement", evidence).text
        second = generator.generate("annual leave entitlement", evidence).text
        assert first == second

    def test_answer_text_is_extracted_verbatim(self, engine: AHRAGEngine) -> None:
        """No sentence is invented: each appears in the cited chunk."""
        evidence = _scored(engine, ["doc-hr-leave-v2::c001"], score=0.9)
        answer = ExtractiveGenerator().generate("annual leave entitlement", evidence)
        source = evidence[0].chunk.text
        normalised_source = " ".join(source.split())
        for line in answer.text.splitlines():
            if not line.strip().startswith("- "):
                continue
            sentence = line.strip()[2:].split(" [")[0].strip()
            assert sentence in normalised_source

    def test_conflicts_are_disclosed_in_the_answer(self, engine: AHRAGEngine) -> None:
        """A detected conflict appears in the generated text."""
        evidence = _scored(
            engine, ["doc-hr-leave-v2::c001", "doc-hr-leave-v1::c002"], score=0.9
        )
        conflicts = ConflictDetector(engine.freshness).detect(evidence)
        answer = ExtractiveGenerator().generate(
            "annual leave entitlement", evidence, conflicts
        )
        assert "Conflicting sources" in answer.text

    def test_no_evidence_produces_no_claims(self) -> None:
        """With an empty pack the generator makes no factual statement."""
        answer = ExtractiveGenerator().generate("anything", [])
        assert answer.citations == []
        assert "no answer can be grounded" in answer.text


class TestCitationVerification:
    """The verifier that polices every backend's output."""

    def test_fabricated_citation_is_stripped(self, engine: AHRAGEngine) -> None:
        """A citation to a non-existent chunk is removed and reported."""
        evidence = _scored(engine, ["doc-hr-leave-v2::c001"])
        text = "Real fact [doc-hr-leave-v2::c001]. Invented [doc-fake::c999]."
        cleaned, citations, stripped = verify_citations(text, evidence)
        assert stripped == ["doc-fake::c999"]
        assert "doc-fake::c999" not in cleaned
        assert [c.chunk_id for c in citations] == ["doc-hr-leave-v2::c001"]

    def test_citation_outside_the_pack_is_stripped(self, engine: AHRAGEngine) -> None:
        """A real chunk that was not served to this user is still removed.

        This is the case that matters for governance: an LLM citing a real but
        filtered chunk must not have that citation displayed.
        """
        evidence = _scored(engine, ["doc-hr-leave-v2::c001"])
        text = "Claim [doc-fin-q3-forecast::c001]."
        cleaned, citations, stripped = verify_citations(text, evidence)
        assert stripped == ["doc-fin-q3-forecast::c001"]
        assert citations == []

    def test_citations_are_ordered_by_first_appearance(
        self, engine: AHRAGEngine
    ) -> None:
        """The displayed citation list matches reading order."""
        evidence = _scored(
            engine, ["doc-hr-leave-v2::c001", "doc-hr-leave-v2::c002"]
        )
        text = "B [doc-hr-leave-v2::c002]. A [doc-hr-leave-v2::c001]."
        _, citations, _ = verify_citations(text, evidence)
        assert [c.chunk_id for c in citations] == [
            "doc-hr-leave-v2::c002",
            "doc-hr-leave-v2::c001",
        ]

    def test_repeated_citation_appears_once(self, engine: AHRAGEngine) -> None:
        """A chunk cited twice is listed once."""
        evidence = _scored(engine, ["doc-hr-leave-v2::c001"])
        text = "A [doc-hr-leave-v2::c001]. B [doc-hr-leave-v2::c001]."
        _, citations, _ = verify_citations(text, evidence)
        assert len(citations) == 1
