"""Access control, freshness, and authority.

These are the tests that back the central governance claim. If any of them
fail, the prototype's core assertion — that an unauthorised chunk cannot
influence an answer — is false.
"""

from __future__ import annotations

import pytest

from ahrag.governance.acl import AccessControl, UnauthorisedChunkError
from ahrag.governance.freshness import FreshnessPolicy
from ahrag.models import Route, ScoredChunk, User
from ahrag.pipeline import AHRAGEngine

from .conftest import TEST_TODAY

RESTRICTED_DOC = "doc-fin-q3-forecast"
ENGINEERING_DOC = "doc-eng-payments-runbook"


class TestAuthorisedScope:
    """The ACL pre-filter that bounds everything downstream."""

    def test_finance_doc_visible_only_to_finance_role(self, engine: AHRAGEngine) -> None:
        """Only the finance role may see the restricted forecast."""
        for user in engine.db.get_users():
            scope = engine.acl.scope_for(user)
            restricted = [
                cid for cid in scope.allowed_chunk_ids if cid.startswith(RESTRICTED_DOC)
            ]
            if "finance" in user.roles:
                assert restricted, f"{user.user_id} should see the finance document"
            else:
                assert not restricted, (
                    f"{user.user_id} (roles={user.roles}) must not see "
                    f"{RESTRICTED_DOC} chunks, but saw {restricted}"
                )

    def test_runbook_hidden_from_non_engineering(self, engine: AHRAGEngine) -> None:
        """A contractor with only the employee role sees no engineering runbook."""
        erin = engine.get_user("erin.contractor")
        scope = engine.acl.scope_for(erin)
        assert not [
            cid for cid in scope.allowed_chunk_ids if cid.startswith(ENGINEERING_DOC)
        ]

    def test_scope_counts_are_consistent(self, engine: AHRAGEngine) -> None:
        """Allowed plus withheld always equals the corpus size."""
        for user in engine.db.get_users():
            scope = engine.acl.scope_for(user)
            assert (
                len(scope.allowed_chunk_ids) + scope.withheld_count
                == scope.total_chunks
                == engine.acl.total_chunks
            )

    def test_withheld_metadata_exposes_types_not_titles(
        self, engine: AHRAGEngine
    ) -> None:
        """Explaining a restriction must not name the restricted documents."""
        alice = engine.get_user("alice.employee")
        scope = engine.acl.scope_for(alice)
        assert scope.withheld_count > 0
        assert "finance" in scope.withheld_doc_types
        titles = {meta.title for meta in engine.db.get_documents()}
        for value in scope.withheld_doc_types:
            assert value not in titles

    def test_user_with_no_roles_gets_empty_scope(self, engine: AHRAGEngine) -> None:
        """A user holding no recognised role may read nothing."""
        nobody = User(user_id="nobody", display_name="Nobody", roles=["unknown-role"])
        scope = engine.acl.scope_for(nobody)
        assert scope.is_empty
        assert scope.restricted_fraction == 1.0


class TestDefenceInDepth:
    """The invariant re-check that should be unreachable in correct operation."""

    def test_assert_authorised_raises_on_unauthorised_chunk(
        self, engine: AHRAGEngine
    ) -> None:
        """Bypassing the pre-filter is detected, not silently tolerated."""
        finance_chunk = next(
            c for c in engine.db.get_chunks() if c.doc_id == RESTRICTED_DOC
        )
        alice = engine.get_user("alice.employee")
        with pytest.raises(UnauthorisedChunkError, match="not readable"):
            engine.acl.assert_authorised(
                [ScoredChunk(chunk=finance_chunk, score=1.0)], alice, stage="test"
            )

    def test_assert_authorised_passes_for_permitted_chunk(
        self, engine: AHRAGEngine
    ) -> None:
        """Authorised material passes the same check without raising."""
        dan = engine.get_user("dan.finance")
        finance_chunk = next(
            c for c in engine.db.get_chunks() if c.doc_id == RESTRICTED_DOC
        )
        engine.acl.assert_authorised([finance_chunk], dan, stage="test")

    def test_empty_allowlist_returns_nothing_not_everything(
        self, engine: AHRAGEngine
    ) -> None:
        """An empty allow-list must search nothing — the fail-safe direction."""
        assert engine.index.sparse.search("leave policy", 10, []) == []
        vector = engine.index.encode_query("leave policy")
        assert engine.index.vectors.search(vector, 10, []) == []


class TestEndToEndACL:
    """Authorisation across the full pipeline, not just the scope calculation."""

    @pytest.mark.parametrize(
        "user_id,query",
        [
            ("alice.employee", "What is the Q3 2026 revenue forecast and gross margin?"),
            ("bob.manager", "How much unallocated contingency does Finance hold?"),
            ("erin.contractor", "What is the remediation for ERR-5041?"),
            ("carol.hr", "What is the forecast gross margin percentage?"),
        ],
    )
    def test_no_restricted_content_reaches_the_user(
        self, engine: AHRAGEngine, user_id: str, query: str
    ) -> None:
        """No unauthorised chunk appears in evidence, citations, or answer text."""
        user = engine.get_user(user_id)
        authorised = set(engine.acl.scope_for(user).allowed_chunk_ids)
        result = engine.answer(query, user_id, write_audit=False)

        for item in result.evidence:
            assert item.chunk_id in authorised, f"leaked {item.chunk_id}"
        for citation in result.citations:
            assert citation.chunk_id in authorised, f"leaked {citation.chunk_id}"

        # The restricted figures themselves must not appear in the answer text.
        for secret in ("48.2", "71.4", "£3.0m", "231"):
            assert secret not in result.answer, (
                f"restricted value {secret!r} surfaced for {user_id}"
            )

    def test_authorised_user_can_read_restricted_content(
        self, engine: AHRAGEngine
    ) -> None:
        """The finance role gets the answer that others are refused.

        Without this, the ACL tests would pass trivially on a system that simply
        never answers anything.
        """
        result = engine.answer(
            "What is the Q3 2026 revenue forecast?", "dan.finance", write_audit=False
        )
        assert not result.abstained
        assert any(e.doc_id == RESTRICTED_DOC for e in result.evidence)
        assert "48.2" in result.answer

    def test_audit_log_records_no_restricted_document(
        self, fresh_engine: AHRAGEngine
    ) -> None:
        """Withheld documents must not appear in the audit trail either."""
        fresh_engine.answer(
            "What is the Q3 2026 revenue forecast and gross margin?",
            "alice.employee",
        )
        records = fresh_engine.db.get_audit_records(limit=5)
        assert records
        for record in records:
            assert RESTRICTED_DOC not in record["documents_considered"]
            assert RESTRICTED_DOC not in record["documents_used"]


class TestFreshness:
    """Freshness and authority preferences."""

    def test_superseded_chunk_is_demoted(self, engine: AHRAGEngine) -> None:
        """A retired policy version scores below its live replacement."""
        policy = FreshnessPolicy(engine.config.evidence, today=TEST_TODAY)
        old = next(c for c in engine.db.get_chunks() if c.doc_id == "doc-hr-leave-v1")
        new = next(c for c in engine.db.get_chunks() if c.doc_id == "doc-hr-leave-v2")
        assert old.is_superseded
        assert not new.is_superseded
        assert policy.freshness_multiplier(old) < policy.freshness_multiplier(new)

    def test_preferred_in_family_picks_current_version(
        self, engine: AHRAGEngine
    ) -> None:
        """The live, higher-authority version wins the preference ordering."""
        policy = FreshnessPolicy(engine.config.evidence, today=TEST_TODAY)
        family = [
            c
            for c in engine.db.get_chunks()
            if c.policy_family == "leave-policy" and c.ordinal == 2
        ]
        assert len(family) >= 2
        preferred = policy.preferred_in_family(family)
        assert preferred is not None
        assert preferred.doc_id == "doc-hr-leave-v2"

    def test_warnings_name_the_superseded_source(self, engine: AHRAGEngine) -> None:
        """A freshness warning identifies the retired document explicitly."""
        policy = FreshnessPolicy(engine.config.evidence, today=TEST_TODAY)
        old = next(c for c in engine.db.get_chunks() if c.doc_id == "doc-hr-leave-v1")
        warnings = policy.warnings_for([old])
        assert warnings and "superseded" in warnings[0].lower()

    def test_authority_multiplier_is_monotonic_and_bounded(self) -> None:
        """Authority nudges ordering without overpowering relevance."""
        from ahrag.config import EvidenceParams

        policy = FreshnessPolicy(EvidenceParams(), today=TEST_TODAY)

        class _Stub:
            authority_score = 1

        low = policy.authority_multiplier(_Stub())  # type: ignore[arg-type]
        _Stub.authority_score = 5
        high = policy.authority_multiplier(_Stub())  # type: ignore[arg-type]
        assert low < high
        assert 0.85 <= low <= 1.0
        assert 1.0 <= high <= 1.15


class TestAccessControlUnit:
    """AccessControl in isolation."""

    def test_rebuild_replaces_snapshot(self, engine: AHRAGEngine) -> None:
        """Rebuilding swaps the corpus snapshot cleanly."""
        acl = AccessControl(engine.db.get_chunks())
        assert acl.total_chunks > 0
        acl.rebuild([])
        assert acl.total_chunks == 0
        user = engine.get_user("carol.hr")
        assert acl.scope_for(user).is_empty

    def test_resolve_authorised_drops_forbidden_ids(self, engine: AHRAGEngine) -> None:
        """Resolving IDs never returns a chunk outside the user's scope."""
        alice = engine.get_user("alice.employee")
        ids = [c.chunk_id for c in engine.db.get_chunks()]
        resolved = engine.acl.resolve_authorised(ids, alice)
        assert resolved
        assert all(not c.chunk_id.startswith(RESTRICTED_DOC) for c in resolved)
