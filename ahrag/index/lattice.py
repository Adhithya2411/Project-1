"""The authorisation lattice: the corpus as a family of subcorpora.

Design position
---------------
``governance/acl.py`` establishes that authorisation is a hard pre-filter: every
retriever is physically restricted to an ``AuthorisedScope``. This module takes
the next step and observes what that implies for *retrieval statistics*.

Under ACL the corpus is not one corpus. It is a lattice of subcorpora indexed by
authorised scope. Yet BM25 IDF, average document length, the TF-IDF vocabulary,
and the LSA basis are all fitted **once, globally**, and only then restricted at
query time. The statistics therefore describe a pool that nobody actually
searches.

That mismatch has two consequences, and the second is the sharper one:

1. **Ranking quality.** A term that is rare corpus-wide but common inside a
   principal's authorised pool receives an inflated IDF, and vice versa. The
   ordering within the authorised pool is computed with the wrong weights.

2. **Information flow.** Because global IDF is a function of every chunk,
   the *scores of authorised chunks depend on the content of unauthorised
   ones*. No restricted text is ever returned, so this is not an access-control
   violation — but it is a **non-interference** violation in the sense of
   Goguen and Meseguer (1982). ``INVENTION_DISCLOSURE.md`` M1 claims an
   unauthorised chunk "cannot influence a route choice, a rerank ordering";
   with globally-fitted statistics, it can. Measured on the seed corpus, top-1
   results flip for authorised users when documents they cannot read change.

An *ACL equivalence class* is a maximal set of principals with identical
authorised scope. Classes, not users, are the unit of specialisation: the count
is bounded by the distinct role-set combinations appearing on documents, which
is small and independent of headcount. On the seed corpus, five users collapse
to four classes.

This module only identifies and describes classes. Building specialised indexes
for them is ``ahrag/index/scoped.py``.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..models import AuthorisedScope, Chunk

logger = logging.getLogger(__name__)

# Signature of the empty scope. Kept explicit because an empty scope is a
# governance-meaningful state (the router abstains on it), not an error.
EMPTY_SIGNATURE = "empty"


def scope_signature(allowed_chunk_ids: Iterable[str]) -> str:
    """Return a stable identifier for an authorised chunk set.

    Two principals share a signature exactly when they may read exactly the
    same chunks — which is precisely when they may share a specialised index.
    The signature is order-insensitive so it does not depend on how the ACL
    layer happened to enumerate the corpus.

    Args:
        allowed_chunk_ids: The authorised chunk IDs. Order is irrelevant.

    Returns:
        A short hex digest, or :data:`EMPTY_SIGNATURE` for an empty set.
    """
    ids = sorted(set(allowed_chunk_ids))
    if not ids:
        return EMPTY_SIGNATURE
    digest = hashlib.blake2b(digest_size=12)
    for chunk_id in ids:
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass(frozen=True)
class ACLClass:
    """One ACL equivalence class: a subcorpus plus the roles that reach it."""

    signature: str
    chunk_ids: frozenset[str]
    doc_ids: frozenset[str]
    role_sets: tuple[frozenset[str], ...] = field(default=())

    @property
    def size(self) -> int:
        """Number of chunks in this subcorpus."""
        return len(self.chunk_ids)

    def contains_all(self, chunk_ids: Iterable[str]) -> bool:
        """True when every ID given is inside this class's subcorpus.

        Used as a purity precondition: a specialised index for this class may
        only ever be asked to rank chunks it was actually fitted on.
        """
        return set(chunk_ids) <= self.chunk_ids


class ScopeLattice:
    """Describes the ACL equivalence classes over a chunk snapshot.

    Two distinct jobs, deliberately kept apart:

    * :meth:`classify` answers "which class is this live scope in?" — the hot
      path, one hash of the allow-list.
    * :meth:`enumerate_role_classes` answers "how many classes could exist?" —
      a static analysis over the corpus used for capacity planning and for the
      reporting in the evaluation harness. It does not need a principal.
    """

    def __init__(self, chunks: Sequence[Chunk] = ()) -> None:
        """Bind to a chunk snapshot. Call :meth:`rebuild` after re-ingestion."""
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self.rebuild(chunks)

    def rebuild(self, chunks: Sequence[Chunk]) -> None:
        """Replace the chunk snapshot."""
        self._chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in self._chunks}

    @property
    def total_chunks(self) -> int:
        """Chunks in the full corpus, across all classes."""
        return len(self._chunks)

    def classify(self, scope: AuthorisedScope) -> ACLClass:
        """Return the ACL class for a live authorised scope.

        Args:
            scope: The scope computed by ``AccessControl.scope_for``.

        Returns:
            The :class:`ACLClass` this scope belongs to. Built from the scope's
            own allow-list, so it is correct even for a principal whose role
            combination was never seen before.
        """
        chunk_ids = frozenset(scope.allowed_chunk_ids)
        doc_ids = frozenset(
            self._by_id[cid].doc_id for cid in chunk_ids if cid in self._by_id
        )
        return ACLClass(
            signature=scope_signature(chunk_ids),
            chunk_ids=chunk_ids,
            doc_ids=doc_ids,
            role_sets=(frozenset(r.lower() for r in scope.roles),),
        )

    def enumerate_role_classes(self) -> list[ACLClass]:
        """Enumerate the classes reachable by single-role principals.

        A static lower bound on lattice width, computed from the corpus alone.
        Real principals hold role *sets*, whose scopes are unions of these, so
        the true number of reachable classes is larger — but this is the figure
        that matters for "does specialisation have anything to work with?"

        Returns:
            One class per distinct role appearing in the corpus, largest first.
        """
        by_role: dict[str, set[str]] = defaultdict(set)
        for chunk in self._chunks:
            for role in chunk.acl_roles:
                by_role[role.lower()].add(chunk.chunk_id)

        classes: list[ACLClass] = []
        for role, chunk_ids in by_role.items():
            frozen = frozenset(chunk_ids)
            classes.append(
                ACLClass(
                    signature=scope_signature(frozen),
                    chunk_ids=frozen,
                    doc_ids=frozenset(
                        self._by_id[cid].doc_id for cid in frozen if cid in self._by_id
                    ),
                    role_sets=(frozenset({role}),),
                )
            )
        classes.sort(key=lambda c: (-c.size, c.signature))
        return classes

    def describe(self, scopes: Sequence[AuthorisedScope]) -> dict[str, object]:
        """Summarise the lattice induced by a set of live scopes.

        Reported by the evaluation harness so a reader can see whether
        specialisation had any structure to exploit. On a corpus where every
        document grants the same role, ``distinct_classes`` collapses to 1 and
        specialisation is provably a no-op — which is a result worth printing
        rather than discovering by surprise.

        Args:
            scopes: Live scopes, typically one per known principal.

        Returns:
            A JSON-serialisable summary.
        """
        classes: dict[str, ACLClass] = {}
        members: dict[str, list[str]] = defaultdict(list)
        for scope in scopes:
            acl_class = self.classify(scope)
            classes.setdefault(acl_class.signature, acl_class)
            members[acl_class.signature].append(scope.user_id)

        sizes = sorted((c.size for c in classes.values()), reverse=True)
        return {
            "total_chunks": self.total_chunks,
            "principals": len(scopes),
            "distinct_classes": len(classes),
            "role_classes": len(self.enumerate_role_classes()),
            "class_sizes": sizes,
            "smallest_class_fraction": (
                round(sizes[-1] / self.total_chunks, 4)
                if sizes and self.total_chunks
                else 0.0
            ),
            "specialisation_is_degenerate": len(classes) <= 1,
            "members": {sig: sorted(users) for sig, users in members.items()},
        }


__all__ = ["ACLClass", "EMPTY_SIGNATURE", "ScopeLattice", "scope_signature"]
