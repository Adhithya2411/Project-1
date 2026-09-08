"""Index-level non-interference under Scope-Pure Index Specialisation.

Why this file exists
--------------------
``tests/test_governance.py`` establishes *access control*: an unauthorised chunk
never reaches an answer. That is a property of the value returned. It says
nothing about whether unauthorised chunks *influenced* the value returned.

``INVENTION_DISCLOSURE.md`` M1 makes the stronger claim — that an unauthorised
chunk "cannot influence a route choice, a rerank ordering, an evidence pack".
That is a **non-interference** claim in the sense of Goguen and Meseguer (1982),
and with globally-fitted retrieval statistics it is false: BM25 IDF and average
document length are computed over the whole corpus, so the scores of authorised
chunks are a function of unauthorised content.

``TestGlobalIndexLeaks`` demonstrates the violation. The remaining classes
assert that specialisation closes it.

The experimental form throughout is the standard non-interference test: hold the
observer's authorised subcorpus fixed, vary everything outside it, and require
the observation to be bit-identical.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ahrag.config import RouterConfig, Settings
from ahrag.db import Database
from ahrag.index.lattice import EMPTY_SIGNATURE, ScopeLattice, scope_signature
from ahrag.index.scoped import DEFAULT_SATURATION, ScopeSpecialisedIndex
from ahrag.index.store import IndexBundle
from ahrag.models import Chunk
from ahrag.pipeline import AHRAGEngine

from .conftest import TEST_TODAY

REPO_ROOT = Path(__file__).resolve().parent.parent

# A contractor holding only `employee` has the narrowest scope in the seed
# corpus (35 of 60 chunks), so it is the observer most exposed to interference
# from the 25 chunks it cannot read.
OBSERVER = "erin.contractor"

# Queries chosen to span the seed corpus's topics. Non-interference must hold
# for every query, so a failure on any one of them is a failure.
PROBE_QUERIES = [
    "how many annual leave days do I get",
    "what is the expense approval limit for travel",
    "remote work eligibility requirements",
    "payment gateway error code retry",
    "quarterly revenue forecast",
    "who approves a policy exception",
]


def _specialised_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Settings with specialisation on, over an isolated database."""
    base: dict[str, object] = {
        "data_dir": tmp_path,
        "db_path": tmp_path / "spis.sqlite3",
        "router_config": REPO_ROOT / "config" / "router.yaml",
        "index_specialisation": True,
        "specialisation_lambda": 0.0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _engine(settings: Settings, config: RouterConfig) -> AHRAGEngine:
    """Seed an engine on its own database."""
    built = AHRAGEngine(
        settings=settings,
        config=config,
        db=Database(settings.db_path),
        today=TEST_TODAY,
    )
    built.seed(reset=True)
    return built


def _rank(bundle_or_view, query: str, allowed: list[str]) -> list[tuple[str, float]]:
    """Return ``(chunk_id, raw_score)`` for a sparse search, best first."""
    return [(cid, raw) for cid, raw, _ in bundle_or_view.sparse.search(query, 10, allowed)]


def _drop_unreadable(chunks: list[Chunk], roles: set[str]) -> list[Chunk]:
    """Return only the chunks a principal with ``roles`` may read."""
    return [c for c in chunks if c.readable_by(roles)]


class TestScopeSignature:
    """The equivalence-class identity function."""

    def test_signature_is_order_insensitive(self) -> None:
        """Identical sets hash identically regardless of enumeration order."""
        assert scope_signature(["b", "a", "c"]) == scope_signature(["c", "a", "b"])

    def test_signature_ignores_duplicates(self) -> None:
        """A repeated ID does not create a distinct class."""
        assert scope_signature(["a", "a", "b"]) == scope_signature(["a", "b"])

    def test_different_sets_differ(self) -> None:
        """Removing a single chunk yields a different class."""
        assert scope_signature(["a", "b"]) != scope_signature(["a"])

    def test_empty_scope_has_its_own_signature(self) -> None:
        """An empty scope is a governance state, not an error."""
        assert scope_signature([]) == EMPTY_SIGNATURE


class TestLattice:
    """The lattice must actually have structure to exploit."""

    def test_seed_corpus_is_not_degenerate(self, engine: AHRAGEngine) -> None:
        """Five seed users collapse to more than one ACL class."""
        lattice = ScopeLattice(engine.db.get_chunks())
        scopes = [engine.acl.scope_for(u) for u in engine.db.get_users()]
        summary = lattice.describe(scopes)
        assert summary["distinct_classes"] > 1, (
            "specialisation is a no-op on a corpus with one ACL class; "
            f"got {summary}"
        )
        assert summary["specialisation_is_degenerate"] is False

    def test_principals_with_equal_scope_share_a_class(
        self, engine: AHRAGEngine
    ) -> None:
        """bob.manager and carol.hr read the same chunks, so they share an index."""
        lattice = ScopeLattice(engine.db.get_chunks())
        bob = lattice.classify(engine.acl.scope_for(engine.get_user("bob.manager")))
        carol = lattice.classify(engine.acl.scope_for(engine.get_user("carol.hr")))
        assert bob.signature == carol.signature

    def test_narrower_scope_is_a_distinct_class(self, engine: AHRAGEngine) -> None:
        """A contractor and an engineer do not share an index."""
        lattice = ScopeLattice(engine.db.get_chunks())
        erin = lattice.classify(engine.acl.scope_for(engine.get_user(OBSERVER)))
        alice = lattice.classify(engine.acl.scope_for(engine.get_user("alice.employee")))
        assert erin.signature != alice.signature
        assert erin.size < alice.size

    def test_class_contains_exactly_its_own_chunks(self, engine: AHRAGEngine) -> None:
        """The purity precondition: a class knows which chunks it may rank."""
        lattice = ScopeLattice(engine.db.get_chunks())
        scope = engine.acl.scope_for(engine.get_user(OBSERVER))
        acl_class = lattice.classify(scope)
        assert acl_class.contains_all(scope.allowed_chunk_ids)
        unreadable = set(c.chunk_id for c in engine.db.get_chunks()) - acl_class.chunk_ids
        assert unreadable, "observer should have unreadable chunks for this to mean anything"
        assert not acl_class.contains_all(unreadable)


class TestGlobalIndexLeaks:
    """The violation this work exists to close.

    These tests *assert the presence of a leak*. They are written to fail if the
    unspecialised path ever becomes non-interfering on its own, because at that
    point the specialisation argument would need revisiting rather than
    silently passing.
    """

    def test_global_idf_depends_on_unreadable_chunks(
        self, engine: AHRAGEngine, settings: Settings
    ) -> None:
        """An authorised chunk's BM25 score changes when unreadable text changes."""
        chunks = engine.db.get_chunks()
        erin = engine.get_user(OBSERVER)
        allowed = engine.acl.scope_for(erin).allowed_chunk_ids

        full = IndexBundle(settings)
        full.build(chunks)
        without = IndexBundle(settings)
        without.build(_drop_unreadable(chunks, erin.role_set))

        differences = []
        for query in PROBE_QUERIES:
            a = dict(_rank(full, query, allowed))
            b = dict(_rank(without, query, allowed))
            for chunk_id, score in a.items():
                other = b.get(chunk_id)
                if other is not None and abs(score - other) > 1e-6:
                    differences.append((query, chunk_id, score, other))

        assert differences, (
            "expected globally-fitted BM25 to leak: authorised chunk scores "
            "should change when unreadable documents are removed"
        )

    def test_global_index_can_reorder_authorised_results(
        self, engine: AHRAGEngine, settings: Settings
    ) -> None:
        """The leak is strong enough to change the observer's ranking, not just scores."""
        chunks = engine.db.get_chunks()
        erin = engine.get_user(OBSERVER)
        allowed = engine.acl.scope_for(erin).allowed_chunk_ids

        full = IndexBundle(settings)
        full.build(chunks)
        without = IndexBundle(settings)
        without.build(_drop_unreadable(chunks, erin.role_set))

        reordered = [
            query
            for query in PROBE_QUERIES
            if [cid for cid, _ in _rank(full, query, allowed)]
            != [cid for cid, _ in _rank(without, query, allowed)]
        ]
        assert reordered, (
            "expected at least one query where unreadable documents change the "
            "observer's result order"
        )


class TestScopePureNonInterference:
    """With specialisation on, unreadable content must have no observable effect."""

    @pytest.fixture
    def observer_scope(self, engine: AHRAGEngine):  # noqa: ANN201
        """The narrowest seed scope, used as the observer throughout."""
        return engine.acl.scope_for(engine.get_user(OBSERVER))

    def test_specialised_index_is_fitted_only_on_its_class(
        self, engine: AHRAGEngine, settings: Settings, observer_scope
    ) -> None:
        """The specialised bundle indexes exactly the authorised chunks."""
        spis = ScopeSpecialisedIndex(
            Settings(**{**settings.model_dump(), "index_specialisation": True}),
            engine.index,
        )
        view = spis.for_scope(observer_scope)
        assert view.specialised
        assert view.chunk_count == len(observer_scope.allowed_chunk_ids)
        assert len(view.sparse) == len(observer_scope.allowed_chunk_ids)

    def test_scores_are_invariant_to_unreadable_content(
        self, engine: AHRAGEngine, settings: Settings, observer_scope
    ) -> None:
        """The core theorem, at the score level.

        Build a specialised index for the observer's class from the full corpus,
        then from a corpus with every unreadable chunk deleted. Because the
        class subcorpus is identical in both, every score must match exactly.
        """
        specialised = Settings(**{**settings.model_dump(), "index_specialisation": True})
        chunks = engine.db.get_chunks()
        erin = engine.get_user(OBSERVER)
        allowed = observer_scope.allowed_chunk_ids

        full = IndexBundle(specialised)
        full.build(chunks)
        without = IndexBundle(specialised)
        without.build(_drop_unreadable(chunks, erin.role_set))

        view_full = ScopeSpecialisedIndex(specialised, full).for_scope(observer_scope)
        view_without = ScopeSpecialisedIndex(specialised, without).for_scope(
            observer_scope
        )

        for query in PROBE_QUERIES:
            a = _rank(view_full, query, allowed)
            b = _rank(view_without, query, allowed)
            assert [cid for cid, _ in a] == [cid for cid, _ in b], (
                f"ranking for {query!r} changed when unreadable chunks were removed"
            )
            for (cid_a, score_a), (cid_b, score_b) in zip(a, b):
                assert cid_a == cid_b
                assert score_a == pytest.approx(score_b, abs=1e-9), (
                    f"score for {cid_a} on {query!r} changed: {score_a} vs {score_b}"
                )

    def test_saturation_constant_is_invariant_to_unreadable_content(
        self, engine: AHRAGEngine, settings: Settings, observer_scope
    ) -> None:
        """Confidence calibration must not depend on unreadable content either.

        Run with calibration explicitly enabled: the constant is the input to
        the router's hard ``min_probe_for_answering`` gate, so a leak here would
        be a leak into a governance decision.
        """
        calibrated = Settings(
            **{
                **settings.model_dump(),
                "index_specialisation": True,
                "specialisation_calibrate_confidence": True,
            }
        )
        chunks = engine.db.get_chunks()
        erin = engine.get_user(OBSERVER)

        full = IndexBundle(calibrated)
        full.build(chunks)
        without = IndexBundle(calibrated)
        without.build(_drop_unreadable(chunks, erin.role_set))

        a = ScopeSpecialisedIndex(calibrated, full).for_scope(observer_scope)
        b = ScopeSpecialisedIndex(calibrated, without).for_scope(observer_scope)
        assert a.saturation == pytest.approx(b.saturation, abs=1e-9)
        assert a.saturation != DEFAULT_SATURATION, (
            "calibration was requested but fell back to the default constant, "
            "so this test would pass trivially"
        )

    def test_route_and_evidence_are_invariant_end_to_end(
        self, tmp_path: Path, config: RouterConfig
    ) -> None:
        """The theorem at the pipeline level: same route, same evidence, same answer.

        Two engines on separate databases: one seeded with the whole corpus, one
        with every document the observer cannot read deleted. The observer's
        full ``answer()`` result must be indistinguishable.
        """
        full_settings = _specialised_settings(tmp_path / "full")
        pruned_settings = _specialised_settings(tmp_path / "pruned")

        full_engine = _engine(full_settings, config)
        pruned_engine = _engine(pruned_settings, config)

        erin = pruned_engine.get_user(OBSERVER)
        for document in list(pruned_engine.db.get_documents()):
            readable = any(
                role.lower() in erin.role_set for role in document.acl_roles
            )
            if not readable:
                pruned_engine.db.delete_document(document.doc_id)
        pruned_engine.refresh_indexes()

        assert pruned_engine.db.count_chunks() < full_engine.db.count_chunks(), (
            "pruning removed nothing, so the test would pass trivially"
        )

        for query in PROBE_QUERIES:
            a = full_engine.answer(query, OBSERVER, write_audit=False)
            b = pruned_engine.answer(query, OBSERVER, write_audit=False)
            assert a.decision.route is b.decision.route, (
                f"route for {query!r} changed: {a.decision.route} vs {b.decision.route}"
            )
            assert [e.chunk_id for e in a.evidence] == [
                e.chunk_id for e in b.evidence
            ], f"evidence for {query!r} changed"
            assert a.abstained is b.abstained
            assert a.answer == b.answer, f"answer text for {query!r} changed"

    def test_probe_confidence_is_invariant_end_to_end(
        self, tmp_path: Path, config: RouterConfig
    ) -> None:
        """Router features derived from retrieval must be invariant too."""
        full_engine = _engine(_specialised_settings(tmp_path / "f2"), config)
        pruned_engine = _engine(_specialised_settings(tmp_path / "p2"), config)

        erin = pruned_engine.get_user(OBSERVER)
        for document in list(pruned_engine.db.get_documents()):
            if not any(role.lower() in erin.role_set for role in document.acl_roles):
                pruned_engine.db.delete_document(document.doc_id)
        pruned_engine.refresh_indexes()

        for query in PROBE_QUERIES:
            a = full_engine.answer(query, OBSERVER, write_audit=False).decision.features
            b = pruned_engine.answer(query, OBSERVER, write_audit=False).decision.features
            assert a.sparse_confidence == pytest.approx(b.sparse_confidence, abs=1e-9)
            assert a.dense_confidence == pytest.approx(b.dense_confidence, abs=1e-9)
            assert a.probe_agreement == pytest.approx(b.probe_agreement, abs=1e-9)


class TestLambdaReopensTheChannel:
    """lambda is an information-flow budget, and the tests should say so."""

    def test_lambda_zero_leaves_local_idf_untouched(
        self, engine: AHRAGEngine, settings: Settings
    ) -> None:
        """The pure setting must not consult the global table at all."""
        chunks = engine.db.get_chunks()
        erin = engine.get_user(OBSERVER)
        subset = _drop_unreadable(chunks, erin.role_set)

        bundle = IndexBundle(settings)
        bundle.build(subset)
        before = bundle.sparse.idf_snapshot()

        global_bundle = IndexBundle(settings)
        global_bundle.build(chunks)
        bundle.sparse.shrink_idf_towards(global_bundle.sparse.idf_snapshot(), 0.0)
        assert bundle.sparse.idf_snapshot() == before

    def test_positive_lambda_moves_idf_towards_global(
        self, engine: AHRAGEngine, settings: Settings
    ) -> None:
        """Any lambda above zero imports information from outside the class."""
        chunks = engine.db.get_chunks()
        erin = engine.get_user(OBSERVER)
        subset = _drop_unreadable(chunks, erin.role_set)

        bundle = IndexBundle(settings)
        bundle.build(subset)
        local = bundle.sparse.idf_snapshot()

        global_bundle = IndexBundle(settings)
        global_bundle.build(chunks)
        reference = global_bundle.sparse.idf_snapshot()

        bundle.sparse.shrink_idf_towards(reference, 0.5)
        blended = bundle.sparse.idf_snapshot()

        moved = [
            term
            for term, value in blended.items()
            if abs(value - local[term]) > 1e-9
        ]
        assert moved, "lambda=0.5 should have changed at least one IDF value"
        for term in moved:
            expected = 0.5 * local[term] + 0.5 * reference[term]
            assert blended[term] == pytest.approx(expected, abs=1e-9)

    def test_shrinkage_never_introduces_out_of_class_vocabulary(
        self, engine: AHRAGEngine, settings: Settings
    ) -> None:
        """A term occurring only in unreadable documents must not gain a score.

        This is the leak in its purest form, so it is blocked even at
        ``lambda=1.0`` where the numeric values are fully global.
        """
        chunks = engine.db.get_chunks()
        erin = engine.get_user(OBSERVER)
        subset = _drop_unreadable(chunks, erin.role_set)

        bundle = IndexBundle(settings)
        bundle.build(subset)
        local_terms = set(bundle.sparse.idf_snapshot())

        global_bundle = IndexBundle(settings)
        global_bundle.build(chunks)
        reference = global_bundle.sparse.idf_snapshot()
        assert set(reference) - local_terms, (
            "the full corpus should contain vocabulary absent from the "
            "observer's subcorpus for this test to mean anything"
        )

        bundle.sparse.shrink_idf_towards(reference, 1.0)
        assert set(bundle.sparse.idf_snapshot()) == local_terms

    def test_lambda_is_validated(self, engine: AHRAGEngine, settings: Settings) -> None:
        """An out-of-range budget is a configuration error, not a clamp."""
        bundle = IndexBundle(settings)
        bundle.build(engine.db.get_chunks())
        with pytest.raises(ValueError, match=r"lam must be in \[0, 1\]"):
            bundle.sparse.shrink_idf_towards({"x": 1.0}, 1.5)
        with pytest.raises(ValueError, match=r"lam must be in \[0, 1\]"):
            bundle.sparse.shrink_idf_towards({"x": 1.0}, -0.1)


class TestSpecialisationDegradesSafely:
    """Operational guarantees: bounded memory, correct fallbacks, ACL preserved."""

    def test_disabled_specialisation_is_a_pass_through(
        self, engine: AHRAGEngine, settings: Settings
    ) -> None:
        """With the flag off, the view is the global bundle and nothing is built."""
        spis = ScopeSpecialisedIndex(settings, engine.index)
        scope = engine.acl.scope_for(engine.get_user(OBSERVER))
        view = spis.for_scope(scope)
        assert not view.specialised
        assert view.saturation == DEFAULT_SATURATION
        assert spis.stats()["builds"] == 0

    def test_empty_scope_never_builds_an_index(
        self, engine: AHRAGEngine, settings: Settings
    ) -> None:
        """A principal with no authorised chunks gets no specialised index."""
        specialised = Settings(**{**settings.model_dump(), "index_specialisation": True})
        spis = ScopeSpecialisedIndex(specialised, engine.index)
        from ahrag.models import AuthorisedScope

        empty = AuthorisedScope(
            user_id="nobody",
            roles=["none"],
            allowed_chunk_ids=[],
            allowed_doc_ids=[],
            total_chunks=engine.db.count_chunks(),
            withheld_count=engine.db.count_chunks(),
        )
        view = spis.for_scope(empty)
        assert not view.specialised
        assert view.signature == EMPTY_SIGNATURE
        assert spis.stats()["builds"] == 0

    def test_cache_is_bounded_and_evicts(
        self, engine: AHRAGEngine, settings: Settings
    ) -> None:
        """A cap of one forces eviction rather than unbounded growth."""
        capped = Settings(
            **{
                **settings.model_dump(),
                "index_specialisation": True,
                "specialisation_max_classes": 1,
            }
        )
        spis = ScopeSpecialisedIndex(capped, engine.index)
        users = ["erin.contractor", "alice.employee", "dan.finance"]
        for user_id in users:
            spis.for_scope(engine.acl.scope_for(engine.get_user(user_id)))
        stats = spis.stats()
        assert stats["cached_classes"] == 1
        assert stats["evictions"] >= 2

    def test_cache_hits_on_repeated_scope(
        self, engine: AHRAGEngine, settings: Settings
    ) -> None:
        """The same class is built once, then served from cache."""
        specialised = Settings(**{**settings.model_dump(), "index_specialisation": True})
        spis = ScopeSpecialisedIndex(specialised, engine.index)
        scope = engine.acl.scope_for(engine.get_user(OBSERVER))
        for _ in range(3):
            spis.for_scope(scope)
        stats = spis.stats()
        assert stats["builds"] == 1
        assert stats["cache_hits"] == 2

    def test_reingestion_invalidates_specialised_indexes(
        self, tmp_path: Path, config: RouterConfig
    ) -> None:
        """A stale per-class index would serve results from a superseded corpus."""
        engine = _engine(_specialised_settings(tmp_path / "inv"), config)
        scope = engine.acl.scope_for(engine.get_user(OBSERVER))
        engine.index.for_scope(scope)
        assert engine.index.specialisation.stats()["cached_classes"] == 1

        engine.refresh_indexes()
        assert engine.index.specialisation.stats()["cached_classes"] == 0

    def test_acl_is_still_enforced_under_specialisation(
        self, tmp_path: Path, config: RouterConfig
    ) -> None:
        """Specialisation must not weaken the property it is built on top of."""
        engine = _engine(_specialised_settings(tmp_path / "acl"), config)
        for query in PROBE_QUERIES:
            result = engine.answer(query, OBSERVER, write_audit=False)
            erin = engine.get_user(OBSERVER)
            for item in result.evidence:
                assert set(item.acl_roles) & erin.role_set, (
                    f"{item.chunk_id} surfaced for {OBSERVER} under specialisation "
                    f"but requires roles {item.acl_roles}"
                )

    def test_stats_report_purity_honestly(
        self, tmp_path: Path, config: RouterConfig
    ) -> None:
        """`fully_pure` must be False whenever lambda reopens the channel."""
        pure = _engine(_specialised_settings(tmp_path / "pure"), config)
        pure.index.for_scope(pure.acl.scope_for(pure.get_user(OBSERVER)))
        assert pure.index.specialisation.stats()["fully_pure"] is True

        leaky = _engine(
            _specialised_settings(tmp_path / "leaky", specialisation_lambda=0.5), config
        )
        leaky.index.for_scope(leaky.acl.scope_for(leaky.get_user(OBSERVER)))
        assert leaky.index.specialisation.stats()["fully_pure"] is False
