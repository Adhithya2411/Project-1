"""Demonstration scenarios: what AHRAG does, shown rather than asserted abstractly.

This file is written to be *read*, not just run. Each test is one scenario a
reviewer, examiner, or new team member would want to see for themselves, and
each prints the observable evidence so that ``pytest -s -k demo`` doubles as a
guided tour of the system.

Run the whole tour:

    python -m pytest tests/test_demonstration.py -s -q

Run one scenario:

    python -m pytest tests/test_demonstration.py -s -k acl_boundary

The scenarios deliberately avoid mocks. Every one goes through
``AHRAGEngine.answer`` on the real seeded corpus with the real ACL layer, so
what is printed is what the system actually did.

Naming: ``demo_NN_*`` so the tour runs in a sensible order under pytest's
file-order collection.
"""

from __future__ import annotations

import pytest

from ahrag.models import Route
from ahrag.pipeline import AHRAGEngine

# The seed corpus's planted restricted document: finance role only.
RESTRICTED_DOC = "doc-fin-q3-forecast"


def show(title: str, lines: list[str]) -> None:
    """Print a labelled block so ``-s`` output reads as a report."""
    width = 74
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    for line in lines:
        print(f"  {line}")


def describe(result) -> list[str]:
    """One-block summary of an ``AnswerResult``."""
    return [
        f"route            : {result.decision.route.value}",
        f"router           : {result.decision.router_version}",
        f"abstained        : {result.abstained}"
        + (f"  ({result.abstention_reason.value})" if result.abstained else ""),
        f"evidence chunks  : {len(result.evidence)}",
        f"documents cited  : "
        f"{sorted({c.chunk_id.split('::')[0] for c in result.citations}) or '—'}",
        f"conflicts        : {len(result.conflicts)}",
        f"latency          : {result.total_latency_s * 1000:.1f} ms",
    ]


class TestDemonstration:
    """The guided tour."""

    # -- 1. The same question, two principals ----------------------------

    def test_demo_01_acl_boundary_same_question_two_users(
        self, engine: AHRAGEngine
    ) -> None:
        """The headline behaviour: authorisation changes the answer, not just a filter.

        `dan.finance` may read the restricted Q3 forecast. `erin.contractor`
        may not. The same question therefore has two different *correct*
        outcomes, and the refusal must not disclose that the document exists.
        """
        question = "What is the Q3 2026 revenue forecast?"

        authorised = engine.answer(question, "dan.finance", write_audit=False)
        denied = engine.answer(question, "erin.contractor", write_audit=False)

        show(
            "1. ACL boundary — identical question, two principals",
            [f"question: {question!r}", ""]
            + ["dan.finance (roles: employee, finance)"]
            + [f"  {line}" for line in describe(authorised)]
            + ["", "erin.contractor (roles: employee)"]
            + [f"  {line}" for line in describe(denied)]
            + [
                "",
                "refusal text shown to the contractor:",
                f"  {denied.answer.splitlines()[0]}",
            ],
        )

        # The authorised principal gets the restricted material.
        assert not authorised.abstained
        assert any(
            c.chunk_id.startswith(RESTRICTED_DOC) for c in authorised.citations
        )

        # The denied principal gets nothing from it, by any route.
        assert denied.abstained
        assert not any(e.doc_id == RESTRICTED_DOC for e in denied.evidence)

        # And the refusal does not leak the existence of the withheld document.
        blob = " ".join(
            [denied.answer, denied.clarifying_question or ""]
        ).lower()
        for giveaway in (RESTRICTED_DOC, "q3 2026 revenue", "forecast and margin"):
            assert giveaway.lower() not in blob, (
                f"the abstention message leaked {giveaway!r}"
            )

    # -- 2. Governance outranks utility ----------------------------------

    def test_demo_02_governance_cannot_be_outbid(self, engine: AHRAGEngine) -> None:
        """A cheap, high-utility route cannot be chosen if governance excluded it.

        This is the structural claim: admissibility is computed *before* any
        utility comparison, so no cost or latency saving can reinstate a route.
        """
        # A bare identifier that appears nowhere in this principal's scope
        # drives probe confidence to zero, which is what trips the hard
        # constraint. Note that the *wrapped* form of the same query
        # ("What does error ERR-99999 mean?") scores 0.55 on its function words
        # alone and stays admissible — the floor is sensitive to phrasing, which
        # is itself worth seeing.
        result = engine.answer("ERR-99999", "erin.contractor", write_audit=False)
        excluded = [u for u in result.decision.utilities if not u.admissible]

        show(
            "2. Governance constraints outrank utility",
            [
                f"chosen route : {result.decision.route.value}",
                f"constraints  : {result.decision.hard_constraints_applied}",
                "",
                "routes excluded before utility was compared:",
            ]
            + [f"  {u.route.value}: {u.rejection_reason}" for u in excluded]
            + [
                "",
                "Because exclusion precedes the argmax, no value of the latency,",
                "cost or quality terms can bring these routes back.",
            ],
        )

        assert result.decision.route is Route.R0
        assert excluded, "expected at least one route excluded on governance grounds"
        chosen = result.decision.utility_of(result.decision.route)
        assert chosen is not None and chosen.admissible

    # -- 3. Conflict disclosure ------------------------------------------

    def test_demo_03_version_conflict_is_disclosed_not_resolved(
        self, engine: AHRAGEngine
    ) -> None:
        """Two live versions of a policy are surfaced, with a stated preference.

        The system does not silently pick one. It answers from the current
        version *and* discloses that a superseded version says something else.
        """
        result = engine.answer(
            "What is the current annual leave entitlement now?", "carol.hr",
            write_audit=False,
        )

        lines = describe(result) + [""]
        for conflict in result.conflicts:
            lines += [
                f"conflict ({conflict.kind}) in family {conflict.policy_family!r}:",
                f"  {conflict.description[:150]}",
                f"  chunks    : {conflict.chunk_ids}",
                f"  preferred : {conflict.preferred_chunk_id}",
                f"  because   : {(conflict.preference_reason or '')[:110]}",
                "",
            ]
        lines.append("answer (first line):")
        lines.append(f"  {result.answer.splitlines()[0][:140]}")

        show("3. Version conflict disclosed, not resolved away", lines)

        assert result.conflicts, "expected the planted version conflict"
        assert any(c.kind == "numeric" for c in result.conflicts), (
            "expected the 22-vs-26-days numeric conflict to be detected"
        )
        # The answer leads with the current figure, not the retired one.
        assert "26" in result.answer

    # -- 4. Freshness ----------------------------------------------------

    def test_demo_04_superseded_source_is_labelled(self, engine: AHRAGEngine) -> None:
        """A retired document may appear, but never without a warning."""
        result = engine.answer(
            "What is the current annual leave entitlement now?", "carol.hr",
            write_audit=False,
        )
        superseded = [e for e in result.evidence if e.is_superseded]

        show(
            "4. Superseded evidence is shown but labelled",
            [
                f"evidence items      : {len(result.evidence)}",
                f"superseded among it : {len(superseded)}",
            ]
            + [f"  {e.chunk_id} (superseded_by {e.superseded_by})" for e in superseded]
            + ["", "freshness warnings:"]
            + [f"  {w[:150]}" for w in result.freshness_warnings],
        )

        if superseded:
            assert result.freshness_warnings, (
                "a superseded chunk was shown with no freshness warning"
            )

    # -- 5. Abstention on an unanswerable question -----------------------

    def test_demo_05_abstains_rather_than_inventing(self, engine: AHRAGEngine) -> None:
        """No support in the corpus produces a typed refusal, not a guess."""
        question = "Which vendor supplies the executive submarine fleet?"
        result = engine.answer(question, "alice.employee", write_audit=False)

        # The honest part of this demonstration: abstention on unanswerable
        # questions is unreliable. Four of these five are equally unsupported by
        # the corpus, and most are answered anyway, because words like "policy"
        # and "business travel" match real documents strongly enough to clear
        # the probe floor and the sufficiency gate.
        probes = []
        for other in (
            "What is the company policy on interplanetary business travel?",
            "How many vacation days does the Mars office grant?",
            "What is the escalation path for a zeppelin incident?",
            "What is the retention period for telepathic communications?",
        ):
            sibling = engine.answer(other, "alice.employee", write_audit=False)
            probes.append(
                f"  {'abstains' if sibling.abstained else 'ANSWERS  '} "
                f"probe={max(sibling.decision.features.sparse_confidence, sibling.decision.features.dense_confidence):.3f}"
                f"  {other[:52]!r}"
            )

        show(
            "5. Abstention instead of hallucination — and where it fails",
            [f"question: {question!r}", ""]
            + describe(result)
            + [
                "",
                f"typed reason     : {result.abstention_reason.value}",
                f"clarifying ask   : {result.clarifying_question or '—'}",
                "",
                "answer:",
                f"  {result.answer.splitlines()[0]}",
                "",
                "KNOWN WEAKNESS — equally unsupported questions:",
            ]
            + probes
            + [
                "",
                "Lexical overlap with real documents is enough to clear both the",
                "probe floor and the sufficiency gate. The gate does not separate",
                "'supported' from 'superficially similar'. See FINDINGS.md §4.7.",
            ],
        )

        assert result.abstained
        assert result.abstention_reason.value != "none"
        assert not result.citations

    # -- 6. Every sentence is attributable -------------------------------

    def test_demo_06_every_citation_resolves_to_authorised_evidence(
        self, engine: AHRAGEngine
    ) -> None:
        """Citations are verified against the evidence pack, not generated freely."""
        result = engine.answer(
            "What is the expense approval limit for travel?", "dan.finance",
            write_audit=False,
        )
        evidence_ids = {e.chunk_id for e in result.evidence}
        user = engine.get_user("dan.finance")

        show(
            "6. Citations are verified, not generated",
            describe(result)
            + ["", "citations:"]
            + [
                f"  {c.chunk_id}  (v{c.version}, authority {c.authority_score})"
                f"  ->  in evidence: {c.chunk_id in evidence_ids}"
                for c in result.citations
            ],
        )

        assert result.citations
        for citation in result.citations:
            assert citation.chunk_id in evidence_ids, (
                f"citation {citation.chunk_id} is not in the evidence pack"
            )
        # And every cited chunk is one this user may read.
        for item in result.evidence:
            assert set(item.acl_roles) & user.role_set

    # -- 7. Routing actually varies --------------------------------------

    def test_demo_07_routing_varies_with_the_question(
        self, engine: AHRAGEngine
    ) -> None:
        """Different question shapes select different retrieval strategies."""
        cases = [
            ("exact identifier", "What does error ERR-5041 mean?", "alice.employee"),
            ("semantic lookup", "Can I work from another country?", "alice.employee"),
            ("comparison", "How does the 2023 leave policy differ from the current one?",
             "carol.hr"),
            ("procedural", "How do I escalate a payment incident?", "alice.employee"),
            ("no scope", "What is the Q3 2026 revenue forecast?", "erin.contractor"),
        ]
        rows = []
        for label, question, user_id in cases:
            result = engine.answer(question, user_id, write_audit=False)
            rows.append(
                f"{label:18s} {result.decision.route.value:3s}  "
                f"conf={result.decision.confidence:.2f}  "
                f"{'abstained' if result.abstained else str(len(result.evidence)) + ' evidence'}"
                f"   {question[:44]!r}"
            )

        show("7. The router selects different routes for different questions", rows)

        routes = {
            engine.answer(q, u, write_audit=False).decision.route for _, q, u in cases
        }
        assert len(routes) >= 2, (
            f"expected the router to vary; it chose only {routes}"
        )

    # -- 8. The audit trail ----------------------------------------------

    def test_demo_08_audit_records_the_decision_not_the_content(
        self, fresh_engine: AHRAGEngine
    ) -> None:
        """A route decision is reconstructable without storing the source text."""
        result = fresh_engine.answer(
            "What is the expense approval limit for travel?", "dan.finance",
            write_audit=True,
        )
        records = fresh_engine.db.get_audit_records(limit=10)
        payload = next(
            (r for r in records if r.get("audit_id") == result.audit_id), None
        )
        assert payload is not None, "the query wrote no audit record"
        text_fields = {
            k: v for k, v in payload.items() if isinstance(v, str) and len(v) > 120
        }

        show(
            "8. Audit under content minimisation",
            [
                f"audit_id            : {payload['audit_id']}",
                f"query_hash          : {payload['query_hash']}",
                f"query_text stored   : {payload['query_text']!r}",
                f"route               : {payload['route']}",
                f"documents considered: {payload['documents_considered']}",
                f"documents used      : {payload['documents_used']}",
                f"citations           : {payload['citations']}",
                f"acl pool / withheld : {payload['acl_pool_size']} / {payload['acl_withheld_count']}",
                "",
                f"long free-text fields present: {list(text_fields) or 'none'}",
                "",
                "The decision is auditable; the source text is not retained.",
            ],
        )

        assert payload["query_hash"]
        assert payload["query_text"] is None, (
            "raw query text was persisted despite verbose_audit being off"
        )
        # No evidence body should be recoverable from the record.
        blob = str(payload).lower()
        for chunk in result.evidence:
            snippet = chunk.text[:60].strip().lower()
            if len(snippet) > 30:
                assert snippet not in blob, "evidence text leaked into the audit record"

    # -- 9. Scope shapes the searchable corpus ---------------------------

    def test_demo_09_authorised_scope_differs_per_principal(
        self, engine: AHRAGEngine
    ) -> None:
        """Each principal searches a different subcorpus. This is the ACL lattice."""
        from ahrag.index.lattice import ScopeLattice

        rows = []
        scopes = []
        for user in engine.db.get_users():
            scope = engine.acl.scope_for(user)
            scopes.append(scope)
            rows.append(
                f"{user.user_id:18s} {len(scope.allowed_chunk_ids):3d}/{scope.total_chunks} chunks  "
                f"withheld={scope.withheld_count:3d}  "
                f"types withheld={scope.withheld_doc_types}"
            )

        lattice = ScopeLattice(engine.db.get_chunks())
        summary = lattice.describe(scopes)
        rows += [
            "",
            f"distinct ACL equivalence classes: {summary['distinct_classes']}",
            f"class sizes (chunks)            : {summary['class_sizes']}",
            "",
            "Two principals share a class only when they may read exactly the",
            "same chunks — which is when they may share a specialised index.",
        ]

        show("9. The authorisation lattice", rows)

        assert summary["distinct_classes"] > 1
        assert not summary["specialisation_is_degenerate"]

    # -- 10. Non-interference --------------------------------------------

    def test_demo_10_unreadable_documents_do_not_influence_results(
        self, tmp_path, config
    ) -> None:
        """With specialisation on, deleting unreadable documents changes nothing.

        The demonstration of the novel contribution. Two corpora — one whole,
        one with every document the contractor cannot read removed — must give
        that contractor byte-identical results.
        """
        from pathlib import Path

        from ahrag.config import Settings
        from ahrag.db import Database

        from .conftest import TEST_TODAY

        repo_root = Path(__file__).resolve().parent.parent

        def build(directory: str, specialised: bool) -> AHRAGEngine:
            settings = Settings(
                data_dir=tmp_path / directory,
                db_path=tmp_path / directory / "demo.sqlite3",
                router_config=repo_root / "config" / "router.yaml",
                embedding_backend="lsa",
                reranker="lexical",
                index_specialisation=specialised,
                specialisation_lambda=0.0,
            )
            built = AHRAGEngine(
                settings=settings, config=config,
                db=Database(settings.db_path), today=TEST_TODAY,
            )
            built.seed(reset=True)
            return built

        question = "quarterly revenue forecast"
        rows = []
        for label, specialised in (("unspecialised", False), ("SPIS", True)):
            whole = build(f"{label}-whole", specialised)
            pruned = build(f"{label}-pruned", specialised)
            erin = pruned.get_user("erin.contractor")
            for document in list(pruned.db.get_documents()):
                if not any(r.lower() in erin.role_set for r in document.acl_roles):
                    pruned.db.delete_document(document.doc_id)
            pruned.refresh_indexes()

            a = whole.answer(question, "erin.contractor", write_audit=False)
            b = pruned.answer(question, "erin.contractor", write_audit=False)
            ids_a = [e.chunk_id for e in a.evidence]
            ids_b = [e.chunk_id for e in b.evidence]
            rows += [
                f"{label}:",
                f"  whole corpus  ({whole.db.count_chunks()} chunks): "
                f"{a.decision.route.value} {ids_a[:3]}",
                f"  pruned corpus ({pruned.db.count_chunks()} chunks): "
                f"{b.decision.route.value} {ids_b[:3]}",
                f"  identical     : {ids_a == ids_b and a.decision.route is b.decision.route}",
                "",
            ]
            if specialised:
                assert ids_a == ids_b, (
                    "SPIS must give identical evidence when unreadable "
                    "documents are removed"
                )
                assert a.decision.route is b.decision.route

        rows += [
            "Removing documents the contractor cannot read leaves the SPIS",
            "result untouched. Without specialisation it need not — the",
            "globally-fitted BM25 IDF is a function of those documents.",
        ]
        show("10. Non-interference: unreadable documents have no influence", rows)
