"""End-to-end demo scenarios.

These are the six behaviours the prototype exists to demonstrate. They are
written as executable acceptance criteria: if one fails, the corresponding claim
in the README is no longer true.
"""

from __future__ import annotations

import pytest

from ahrag.models import AbstentionReason, Route
from ahrag.pipeline import AHRAGEngine, UnknownUserError


class TestScenario1ExactErrorCode:
    """Scenario 1: exact identifier lookup routes to sparse retrieval."""

    def test_error_code_routes_to_r1(self, engine: AHRAGEngine) -> None:
        """An error-code question selects R1 and answers from the runbook."""
        result = engine.answer(
            "What is the remediation for ERR-5041?", "alice.employee", write_audit=False
        )
        assert result.decision.route is Route.R1
        assert not result.abstained
        assert any(
            e.doc_id == "doc-eng-payments-runbook" for e in result.evidence
        )

    def test_r1_reason_names_the_identifier(self, engine: AHRAGEngine) -> None:
        """The explanation cites the identifier that drove the choice."""
        result = engine.answer(
            "What is the remediation for ERR-5041?", "alice.employee", write_audit=False
        )
        assert any("ERR-5041" in reason for reason in result.decision.reasons)

    def test_answer_contains_the_remediation(self, engine: AHRAGEngine) -> None:
        """The answer carries real content from the runbook, with citations."""
        result = engine.answer(
            "What is the remediation for ERR-5041?", "alice.employee", write_audit=False
        )
        assert result.citations
        assert "ERR-5041" in result.answer


class TestScenario2ConceptualPolicy:
    """Scenario 2: conceptual questions route to a semantic-capable route."""

    def test_conceptual_question_routes_to_r2_or_r3(self, engine: AHRAGEngine) -> None:
        """A conceptual policy question uses dense or hybrid retrieval."""
        result = engine.answer(
            "Why does the company use a hybrid working model and what is "
            "expected of staff?",
            "alice.employee",
            write_audit=False,
        )
        assert result.decision.route in (Route.R2, Route.R3)
        assert not result.abstained
        assert any(e.doc_id == "doc-hr-remote" for e in result.evidence)

    def test_sparse_route_is_rejected_with_a_reason(self, engine: AHRAGEngine) -> None:
        """R1 scores below the winner on a query with no exact target."""
        result = engine.answer(
            "Why does the company use a hybrid working model and what is "
            "expected of staff?",
            "alice.employee",
            write_audit=False,
        )
        chosen = result.decision.utility_of(result.decision.route)
        sparse = result.decision.utility_of(Route.R1)
        assert chosen and sparse and chosen.utility > sparse.utility


class TestScenario3MultiDocumentComparison:
    """Scenario 3: comparison questions route to decomposed iterative hybrid."""

    QUERY = (
        "Compare the annual leave carry-over rules between the current policy "
        "and the 2023 edition"
    )

    def test_comparison_routes_to_r4(self, engine: AHRAGEngine) -> None:
        """A cross-version comparison selects R4."""
        result = engine.answer(self.QUERY, "carol.hr", write_audit=False)
        assert result.decision.route is Route.R4

    def test_r4_shows_its_subqueries(self, engine: AHRAGEngine) -> None:
        """The decomposition is visible, not hidden."""
        result = engine.answer(self.QUERY, "carol.hr", write_audit=False)
        assert len(result.trace.subqueries) > 1
        assert result.trace.iterations > 1

    def test_comparison_draws_on_both_versions(self, engine: AHRAGEngine) -> None:
        """Evidence spans both policy editions."""
        result = engine.answer(self.QUERY, "carol.hr", write_audit=False)
        doc_ids = {e.doc_id for e in result.evidence}
        assert "doc-hr-leave-v2" in doc_ids
        assert "doc-hr-leave-v1" in doc_ids


class TestScenario4RestrictedFinance:
    """Scenario 4: a user without the finance role gets abstention, no leakage."""

    QUERY = "What is the Q3 2026 revenue forecast and gross margin?"

    def test_unauthorised_user_abstains(self, engine: AHRAGEngine) -> None:
        """The answer is refused rather than partially answered."""
        result = engine.answer(self.QUERY, "alice.employee", write_audit=False)
        assert result.abstained
        assert result.abstention_reason is not AbstentionReason.NONE

    def test_no_restricted_figures_leak(self, engine: AHRAGEngine) -> None:
        """None of the restricted numbers appear anywhere in the response."""
        result = engine.answer(self.QUERY, "alice.employee", write_audit=False)
        blob = result.answer + str([e.text for e in result.evidence])
        for secret in ("48.2", "71.4", "£3.0m", "71.4%"):
            assert secret not in blob

    def test_refusal_does_not_reveal_the_document_exists(
        self, engine: AHRAGEngine
    ) -> None:
        """The refusal must not name the withheld document."""
        result = engine.answer(self.QUERY, "alice.employee", write_audit=False)
        message = result.answer + (result.clarifying_question or "")
        assert "Q3 2026 Revenue Forecast" not in message
        assert "RESTRICTED" not in message

    def test_finance_user_gets_the_answer(self, engine: AHRAGEngine) -> None:
        """The control case: authorisation changes the outcome."""
        result = engine.answer(self.QUERY, "dan.finance", write_audit=False)
        assert not result.abstained
        assert "48.2" in result.answer


class TestScenario5ConflictingVersions:
    """Scenario 5: conflicting policy versions are disclosed, not resolved."""

    QUERY = "What is the current annual leave entitlement now?"

    def test_conflict_is_detected(self, engine: AHRAGEngine) -> None:
        """The live/retired version split is surfaced as a conflict."""
        result = engine.answer(self.QUERY, "carol.hr", write_audit=False)
        assert result.conflicts
        assert any(c.policy_family == "leave-policy" for c in result.conflicts)

    def test_current_version_is_preferred(self, engine: AHRAGEngine) -> None:
        """The answer leads with the live policy's figure."""
        result = engine.answer(self.QUERY, "carol.hr", write_audit=False)
        assert not result.abstained
        assert "26 days" in result.answer
        assert result.evidence[0].doc_id == "doc-hr-leave-v2"

    def test_both_versions_remain_visible(self, engine: AHRAGEngine) -> None:
        """The superseded version is named rather than silently discarded."""
        result = engine.answer(self.QUERY, "carol.hr", write_audit=False)
        conflict = next(c for c in result.conflicts)
        assert len(conflict.chunk_ids) >= 2
        assert conflict.preferred_chunk_id
        assert conflict.preference_reason

    def test_conflict_appears_in_the_answer_text(self, engine: AHRAGEngine) -> None:
        """Disclosure reaches the user, not just the data structure."""
        result = engine.answer(self.QUERY, "carol.hr", write_audit=False)
        assert "Conflicting sources" in result.answer

    def test_freshness_flagged_in_router_reasons(self, engine: AHRAGEngine) -> None:
        """The router records that it treated the query as freshness-sensitive."""
        result = engine.answer(self.QUERY, "carol.hr", write_audit=False)
        assert result.decision.features.freshness_required
        assert any("freshness" in r.lower() for r in result.decision.reasons)


class TestScenario6UnsupportedQuestion:
    """Scenario 6: a question the corpus cannot answer produces a refusal."""

    def test_out_of_domain_question_abstains(self, engine: AHRAGEngine) -> None:
        """A question with no corpus support is refused."""
        result = engine.answer(
            "What is the airspeed velocity of an unladen swallow?",
            "erin.contractor",
            write_audit=False,
        )
        assert result.abstained
        assert not result.citations

    def test_refusal_explains_itself(self, engine: AHRAGEngine) -> None:
        """The refusal states a reason rather than failing silently."""
        result = engine.answer(
            "How do I configure the Kubernetes ingress controller for the mail "
            "gateway?",
            "alice.employee",
            write_audit=False,
        )
        assert result.abstained
        assert result.sufficiency.message
        assert result.abstention_reason is not AbstentionReason.NONE

    def test_never_answers_from_model_memory(self, engine: AHRAGEngine) -> None:
        """A well-known external fact is not answered from parametric knowledge."""
        result = engine.answer(
            "What is the capital city of France?", "alice.employee", write_audit=False
        )
        assert result.abstained or not result.citations
        assert "Paris" not in result.answer


class TestPipelineInvariants:
    """Properties that must hold for every query, on every route."""

    QUERIES = [
        ("alice.employee", "What is the remediation for ERR-5041?"),
        ("bob.manager", "What is the expenses submission deadline?"),
        ("carol.hr", "What is the current annual leave entitlement now?"),
        ("dan.finance", "What is the Q3 revenue forecast?"),
        ("erin.contractor", "What is the hybrid attendance expectation?"),
        ("alice.employee", "Compare the old and new leave carry-over rules"),
    ]

    @pytest.mark.parametrize("user_id,query", QUERIES)
    def test_citations_are_always_grounded(
        self, engine: AHRAGEngine, user_id: str, query: str
    ) -> None:
        """Every citation resolves to a chunk in the served evidence pack."""
        result = engine.answer(query, user_id, write_audit=False)
        served = {e.chunk_id for e in result.evidence}
        for citation in result.citations:
            assert citation.chunk_id in served

    @pytest.mark.parametrize("user_id,query", QUERIES)
    def test_abstention_means_no_citations(
        self, engine: AHRAGEngine, user_id: str, query: str
    ) -> None:
        """An abstention never carries factual citations."""
        result = engine.answer(query, user_id, write_audit=False)
        if result.abstained:
            assert not result.citations

    @pytest.mark.parametrize("user_id,query", QUERIES)
    def test_decision_is_always_explained(
        self, engine: AHRAGEngine, user_id: str, query: str
    ) -> None:
        """Every decision carries reasons and a full utility table."""
        result = engine.answer(query, user_id, write_audit=False)
        assert result.decision.reasons
        assert len(result.decision.utilities) == len(Route)
        assert 0.0 <= result.decision.confidence <= 1.0

    def test_r0_never_generates_a_factual_answer(self, engine: AHRAGEngine) -> None:
        """When R0 is chosen the response is always an abstention."""
        result = engine.answer(
            "What is the Q3 2026 revenue forecast and gross margin?",
            "alice.employee",
            write_audit=False,
        )
        if result.decision.route is Route.R0:
            assert result.abstained
            assert not result.citations

    def test_empty_query_is_rejected(self, engine: AHRAGEngine) -> None:
        """An empty query is a caller error, not an empty answer."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.answer("   ", "alice.employee")

    def test_unknown_user_is_rejected(self, engine: AHRAGEngine) -> None:
        """An unknown user cannot be served a default scope."""
        with pytest.raises(UnknownUserError):
            engine.answer("anything", "mallory.attacker")

    def test_results_are_reproducible(self, engine: AHRAGEngine) -> None:
        """The same query twice gives the same route, evidence, and answer."""
        first = engine.answer("What is the remediation for ERR-5041?", "alice.employee", write_audit=False)
        second = engine.answer("What is the remediation for ERR-5041?", "alice.employee", write_audit=False)
        assert first.decision.route is second.decision.route
        assert [e.chunk_id for e in first.evidence] == [
            e.chunk_id for e in second.evidence
        ]
        assert first.answer == second.answer


class TestAuditTrail:
    """Audit records: complete on the operational side, minimal on content."""

    def test_record_is_written(self, fresh_engine: AHRAGEngine) -> None:
        """A query produces exactly one audit record, linked to the result."""
        result = fresh_engine.answer("What is ERR-5041?", "alice.employee")
        records = fresh_engine.db.get_audit_records(limit=10)
        assert result.audit_id
        assert any(r["audit_id"] == result.audit_id for r in records)

    def test_raw_query_text_is_not_stored_by_default(
        self, fresh_engine: AHRAGEngine
    ) -> None:
        """Only a hash of the query is persisted under the default policy."""
        query = "What is the Q3 2026 revenue forecast?"
        fresh_engine.answer(query, "alice.employee")
        record = fresh_engine.db.get_audit_records(limit=1)[0]
        assert record["query_text"] is None
        assert len(record["query_hash"]) == 64
        assert record["verbose"] == 0

    def test_same_query_hashes_consistently(self, fresh_engine: AHRAGEngine) -> None:
        """Hashes are correlatable across records so frequency is answerable."""
        fresh_engine.answer("What is ERR-5041?", "alice.employee")
        fresh_engine.answer("what is err-5041?  ", "alice.employee")
        records = fresh_engine.db.get_audit_records(limit=2)
        assert records[0]["query_hash"] == records[1]["query_hash"]

    def test_verbose_mode_stores_raw_text(self, tmp_path, config) -> None:
        """The debugging opt-in works and is flagged on the record."""
        from ahrag.config import Settings
        from ahrag.db import Database
        from .conftest import REPO_ROOT, TEST_TODAY

        verbose_settings = Settings(
            data_dir=tmp_path,
            db_path=tmp_path / "verbose.sqlite3",
            router_config=REPO_ROOT / "config" / "router.yaml",
            verbose_audit=True,
        )
        verbose_engine = AHRAGEngine(
            settings=verbose_settings,
            config=config,
            db=Database(verbose_settings.db_path),
            today=TEST_TODAY,
        )
        verbose_engine.seed(reset=True)
        verbose_engine.answer("What is ERR-5041?", "alice.employee")
        record = verbose_engine.db.get_audit_records(limit=1)[0]
        assert record["query_text"] == "What is ERR-5041?"
        assert record["verbose"] == 1

    def test_record_captures_the_full_decision(self, fresh_engine: AHRAGEngine) -> None:
        """Route, features, utilities, timings, and ACL counts are all recorded."""
        fresh_engine.answer("What is the remediation for ERR-5041?", "alice.employee")
        record = fresh_engine.db.get_audit_records(limit=1)[0]
        assert record["route"]
        assert record["router_version"]
        assert record["features"]
        assert len(record["candidate_utilities"]) == len(Route)
        assert record["timings_ms"]
        assert record["acl_pool_size"] > 0
        assert record["documents_considered"]

    def test_abstention_reason_is_recorded(self, fresh_engine: AHRAGEngine) -> None:
        """A refusal is auditable, including why it happened."""
        fresh_engine.answer(
            "What is the Q3 2026 revenue forecast and gross margin?", "alice.employee"
        )
        record = fresh_engine.db.get_audit_records(limit=1)[0]
        assert record["abstained"] is True
        assert record["abstention_reason"] != "none"

    def test_evidence_text_is_never_logged(self, fresh_engine: AHRAGEngine) -> None:
        """Document identifiers are logged; document content is not."""
        fresh_engine.answer("What is the current annual leave entitlement?", "carol.hr")
        record = fresh_engine.db.get_audit_records(limit=1)[0]
        blob = str(record)
        assert "Permanent full-time employees accrue" not in blob
