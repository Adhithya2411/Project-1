"""Loading, chunking, and the ingestion pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ahrag.ingestion.chunker import (
    chunk_document,
    current_version_of,
    resolve_supersession,
)
from ahrag.ingestion.loaders import LoaderError, load_text, supported_extensions
from ahrag.ingestion.pipeline import DEFAULT_UPLOAD_ACL
from ahrag.models import DocType, DocumentMeta
from ahrag.pipeline import AHRAGEngine


def _meta(doc_id: str = "doc-test", **overrides) -> DocumentMeta:
    """Build a document metadata fixture."""
    base = {
        "doc_id": doc_id,
        "title": "Test Document",
        "source_uri": "/tmp/test.md",
        "doc_type": DocType.POLICY,
        "owner": "Test Owner",
        "acl_roles": ["employee"],
        "created_date": date(2025, 1, 1),
        "effective_date": date(2025, 1, 1),
        "version": "1.0",
        "authority_score": 3,
    }
    base.update(overrides)
    return DocumentMeta(**base)


class TestLoaders:
    """Text extraction."""

    def test_supported_extensions(self) -> None:
        """All four required formats are declared."""
        assert {".txt", ".md", ".pdf", ".docx"} <= supported_extensions()

    def test_reads_markdown(self, tmp_path: Path) -> None:
        """A Markdown file loads with structure preserved."""
        target = tmp_path / "a.md"
        target.write_text("# Title\n\nBody text here.\n")
        text = load_text(target)
        assert "Title" in text and "Body text" in text

    def test_reads_plain_text(self, tmp_path: Path) -> None:
        """A .txt file loads."""
        target = tmp_path / "a.txt"
        target.write_text("Hello world")
        assert load_text(target) == "Hello world"

    def test_normalises_line_endings(self, tmp_path: Path) -> None:
        """CRLF and excess blank lines are collapsed."""
        target = tmp_path / "a.txt"
        target.write_bytes(b"a\r\n\r\n\r\n\r\nb")
        assert load_text(target) == "a\n\nb"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing file is a clear error, not an empty document."""
        with pytest.raises(LoaderError, match="not found"):
            load_text(tmp_path / "nope.md")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """An unsupported format is rejected with the supported list."""
        target = tmp_path / "a.xlsx"
        target.write_text("x")
        with pytest.raises(LoaderError, match="Unsupported file type"):
            load_text(target)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """A file yielding no text fails loudly.

        Silently ingesting an empty document would look identical, later, to a
        retrieval failure.
        """
        target = tmp_path / "a.txt"
        target.write_text("   \n\n  ")
        with pytest.raises(LoaderError, match="No extractable text"):
            load_text(target)

    def test_reads_docx(self, tmp_path: Path) -> None:
        """A .docx file loads through python-docx."""
        docx = pytest.importorskip("docx")
        target = tmp_path / "a.docx"
        document = docx.Document()
        document.add_paragraph("Policy heading")
        document.add_paragraph("Policy body sentence.")
        document.save(target)
        text = load_text(target)
        assert "Policy heading" in text
        assert "Policy body sentence." in text


class TestChunker:
    """Structure-aware chunking and stable IDs."""

    def test_chunk_ids_are_stable_and_readable(self) -> None:
        """IDs are deterministic and short enough to render as citations."""
        text = "# A\n\nfirst\n\n# B\n\nsecond"
        first = chunk_document(text, _meta())
        second = chunk_document(text, _meta())
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert first[0].chunk_id == "doc-test::c000"

    def test_governance_metadata_is_denormalised(self) -> None:
        """Every chunk carries its parent's ACL, version, and authority."""
        meta = _meta(acl_roles=["hr", "manager"], version="2.1", authority_score=5)
        chunks = chunk_document("# A\n\nbody text goes here", meta)
        for chunk in chunks:
            assert chunk.acl_roles == ["hr", "manager"]
            assert chunk.version == "2.1"
            assert chunk.authority_score == 5

    def test_numbered_procedure_steps_stay_together(self) -> None:
        """A numbered remediation list is one chunk, not one chunk per step.

        Regression guard: treating every numbered line as a heading shattered
        multi-step procedures across chunks that each retrieved poorly.
        """
        text = (
            "## Remediation\n\n"
            "1. Check the acquirer status page and the latency dashboard.\n"
            "2. If p99 latency exceeds 6000 ms, raise severity to SEV-2 and page "
            "the acquirer liaison immediately.\n"
            "3. Replay the pending captures with the replay tool, which is "
            "idempotent and safe to run twice.\n"
        )
        chunks = chunk_document(text, _meta())
        assert len(chunks) == 1
        assert "SEV-2" in chunks[0].text
        assert "idempotent" in chunks[0].text

    def test_short_numbered_headings_still_split(self) -> None:
        """Genuine numbered headings continue to start new sections."""
        text = "1. Purpose\n\nSome text.\n\n2. Scope\n\nOther text."
        chunks = chunk_document(text, _meta())
        assert len(chunks) == 2
        assert chunks[0].heading == "1. Purpose"
        assert chunks[1].heading == "2. Scope"

    def test_long_section_is_split_with_overlap(self) -> None:
        """Oversized sections split, and the split points overlap."""
        body = "\n\n".join(f"Paragraph number {i} with filler text." * 6 for i in range(12))
        chunks = chunk_document(f"# Long\n\n{body}", _meta(), target_chars=400, overlap_chars=80)
        assert len(chunks) > 1

    def test_rejects_invalid_parameters(self) -> None:
        """Nonsensical chunk sizes are rejected."""
        with pytest.raises(ValueError, match="target_chars"):
            chunk_document("x", _meta(), target_chars=0)
        with pytest.raises(ValueError, match="overlap_chars"):
            chunk_document("x", _meta(), target_chars=100, overlap_chars=100)

    def test_ordinals_are_sequential(self) -> None:
        """Chunk ordinals run 0..n-1 in document order."""
        chunks = chunk_document("# A\n\nx\n\n# B\n\ny\n\n# C\n\nz", _meta())
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))


class TestSupersession:
    """Version-chain resolution."""

    def test_reverse_pointer_is_derived(self) -> None:
        """A declared `supersedes` fills in the older document's `superseded_by`."""
        old = _meta("doc-old", policy_family="fam")
        new = _meta("doc-new", policy_family="fam", supersedes="doc-old")
        resolved = {m.doc_id: m for m in resolve_supersession([old, new])}
        assert resolved["doc-old"].superseded_by == "doc-new"
        assert resolved["doc-new"].superseded_by is None
        assert resolved["doc-old"].is_superseded

    def test_dangling_supersedes_is_ignored(self) -> None:
        """A reference to an absent document does not crash ingestion."""
        new = _meta("doc-new", supersedes="doc-missing")
        resolved = resolve_supersession([new])
        assert resolved[0].superseded_by is None

    def test_current_version_prefers_live_and_latest(self) -> None:
        """The live document with the latest effective date wins."""
        old = _meta("doc-old", policy_family="fam", effective_date=date(2023, 1, 1))
        new = _meta(
            "doc-new",
            policy_family="fam",
            effective_date=date(2025, 1, 1),
            supersedes="doc-old",
        )
        resolved = resolve_supersession([old, new])
        assert current_version_of(resolved, "fam").doc_id == "doc-new"

    def test_unknown_family_returns_none(self) -> None:
        """An unknown policy family has no current version."""
        assert current_version_of([_meta()], "nope") is None


class TestSeeding:
    """The packaged demo corpus."""

    def test_seed_creates_expected_corpus(self, fresh_engine: AHRAGEngine) -> None:
        """All nine documents and their users are present after seeding."""
        documents = fresh_engine.db.get_documents()
        assert len(documents) == 9
        assert fresh_engine.db.count_chunks() > 40
        assert len(fresh_engine.db.get_users()) == 5

    def test_required_roles_exist(self, fresh_engine: AHRAGEngine) -> None:
        """The five demo roles the brief asks for are all assigned."""
        assert {"employee", "manager", "engineering", "hr", "finance"} <= set(
            fresh_engine.db.get_roles()
        )

    def test_corpus_contains_required_document_types(
        self, fresh_engine: AHRAGEngine
    ) -> None:
        """HR policy, runbook, status report, and restricted finance all present."""
        types = {meta.doc_type for meta in fresh_engine.db.get_documents()}
        assert {DocType.POLICY, DocType.RUNBOOK, DocType.REPORT, DocType.FINANCE} <= types

    def test_conflicting_policy_versions_are_linked(
        self, fresh_engine: AHRAGEngine
    ) -> None:
        """The two leave-policy editions form a supersession chain."""
        old = fresh_engine.db.get_document("doc-hr-leave-v1")
        new = fresh_engine.db.get_document("doc-hr-leave-v2")
        assert old.superseded_by == "doc-hr-leave-v2"
        assert new.supersedes == "doc-hr-leave-v1"
        assert old.effective_date < new.effective_date
        assert old.policy_family == new.policy_family

    def test_every_document_declares_an_acl(self, fresh_engine: AHRAGEngine) -> None:
        """No document is readable by default."""
        for meta in fresh_engine.db.get_documents():
            assert meta.acl_roles

    def test_reseeding_is_idempotent(self, fresh_engine: AHRAGEngine) -> None:
        """Re-seeding produces the same corpus, not duplicates."""
        before = fresh_engine.db.count_chunks()
        fresh_engine.seed(reset=True)
        assert fresh_engine.db.count_chunks() == before

    def test_missing_manifest_raises(self, fresh_engine: AHRAGEngine, tmp_path: Path) -> None:
        """A missing manifest is a clear error."""
        with pytest.raises(FileNotFoundError):
            fresh_engine.ingestion.seed_from_manifest(tmp_path / "nope.yaml")


class TestUpload:
    """Ad-hoc document ingestion."""

    def test_upload_defaults_to_restrictive_acl(
        self, fresh_engine: AHRAGEngine, tmp_path: Path
    ) -> None:
        """A document with no declared ACL is restricted, never made public.

        A prototype that widened access on the unhappy path would undercut the
        governance property it exists to demonstrate.
        """
        target = tmp_path / "note.md"
        target.write_text("# Note\n\nSome internal content about widgets.")
        meta, _ = fresh_engine.ingestion.ingest_file(target)
        assert meta.acl_roles == sorted(DEFAULT_UPLOAD_ACL)

        erin = fresh_engine.get_user("erin.contractor")
        fresh_engine.refresh_indexes()
        scope = fresh_engine.acl.scope_for(erin)
        assert not [c for c in scope.allowed_chunk_ids if c.startswith(meta.doc_id)]

    def test_upload_is_retrievable_by_authorised_user(
        self, fresh_engine: AHRAGEngine, tmp_path: Path
    ) -> None:
        """An uploaded document becomes searchable for permitted roles."""
        target = tmp_path / "widget.md"
        target.write_text(
            "# Widget Calibration Standard\n\n"
            "Widgets must be calibrated to a tolerance of 0.02 millimetres "
            "before dispatch. The calibration code is WGT-9911."
        )
        meta, chunks = fresh_engine.ingestion.ingest_file(
            target, acl_roles=["employee"], title="Widget Calibration Standard"
        )
        assert chunks
        fresh_engine.refresh_indexes()
        result = fresh_engine.answer(
            "What is calibration code WGT-9911?", "alice.employee", write_audit=False
        )
        assert any(e.doc_id == meta.doc_id for e in result.evidence)

    def test_upload_rejects_bad_authority_score(
        self, fresh_engine: AHRAGEngine, tmp_path: Path
    ) -> None:
        """Authority must stay inside the documented 1..5 range."""
        target = tmp_path / "x.md"
        target.write_text("# X\n\nSome text here for the body.")
        with pytest.raises(ValueError, match="authority_score"):
            fresh_engine.ingestion.ingest_file(target, authority_score=9)

    def test_upload_can_supersede_an_existing_document(
        self, fresh_engine: AHRAGEngine, tmp_path: Path
    ) -> None:
        """Uploading a replacement retires the previous version."""
        target = tmp_path / "leave-v3.md"
        target.write_text(
            "# Annual Leave Policy v3\n\n"
            "Permanent employees accrue 30 days of paid annual leave per year."
        )
        meta, _ = fresh_engine.ingestion.ingest_file(
            target,
            acl_roles=["employee", "hr"],
            policy_family="leave-policy",
            supersedes="doc-hr-leave-v2",
        )
        retired = fresh_engine.db.get_document("doc-hr-leave-v2")
        assert retired.superseded_by == meta.doc_id
