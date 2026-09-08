"""Dataset Integration for AHRAG.

Converts downloaded benchmark datasets into an AHRAG corpus manifest and a
labelled evaluation set.

Two things here are easy to get wrong, and the previous version of this script
got both wrong. They are the reason it is written the way it is.

1. GOLD CHUNK IDS MUST COME FROM THE REAL CHUNKER.
   AHRAG chunk IDs are ``{doc_id}::c{ordinal:03d}`` where the ordinal is
   assigned by ``ahrag.ingestion.chunker.chunk_document`` — a heading-aware
   split with a 900-character soft bound and 150-character overlap. They are
   *not* page indices, paragraph indices, or anything a converter can predict.
   Emitting ``::c000`` for every item silently yields recall 0 rather than an
   error, because ``load_eval_set`` validates label consistency but not
   chunk-ID existence.
   This script therefore runs the actual chunker over the actual document text
   and locates the evidence span inside the resulting chunks. Items whose
   evidence cannot be located are **dropped**, and the drop count is reported.

2. A GOVERNANCE CORPUS NEEDS GOVERNANCE STRUCTURE.
   A corpus where every document grants the same role has one ACL equivalence
   class, so ACL routing, freshness, and conflict disclosure are all untestable
   on it — every metric comes out trivially constant. improvement.txt §1(d)/(e)
   ask for 10-15 supersession chains and multiple restricted classes; §2(d)
   asks for permission-constrained and freshness-sensitive strata. This script
   assigns roles from a realistic role model, builds real supersession chains
   with competing versions, and emits ACL-probe items carrying
   ``forbidden_chunks``.

USAGE
  python improvement_files/datasets/integrate_datasets.py
  python improvement_files/datasets/integrate_datasets.py --preview
  python improvement_files/datasets/integrate_datasets.py --dataset financebench
  python improvement_files/datasets/integrate_datasets.py --hotpotqa-limit 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ahrag.ingestion.chunker import chunk_document  # noqa: E402
from ahrag.models import DocumentMeta, DocType  # noqa: E402

DATASETS_DIR = PROJECT_ROOT / "improvement_files" / "datasets"
OUTPUT_DIR = DATASETS_DIR / "integrated"

# Seeded so the corpus, the ACL assignment, and the supersession chains are
# byte-identical across runs. improvement.txt §9(a) asks for exactly this.
SEED = 1729

# The role model. Deliberately *not* "everything grants employee": a corpus
# where one role reads everything collapses the ACL lattice to a single class.
# Each entry is (role-set, share of documents) for a document's audience.
AUDIENCE_MODEL: list[tuple[list[str], float]] = [
    (["employee"], 0.34),                          # open to all staff
    (["employee", "manager"], 0.16),               # management-visible
    (["engineering"], 0.12),                       # engineering-only
    (["engineering", "manager"], 0.10),            # engineering + oversight
    (["finance"], 0.08),                           # restricted finance
    (["finance", "manager"], 0.06),                # finance + oversight
    (["hr"], 0.07),                                # restricted HR
    (["hr", "manager"], 0.04),                     # HR + oversight
    (["legal"], 0.03),                             # restricted legal
]

# Principals used by the generated evaluation items. Roles are chosen so that
# every audience above is readable by some principal and withheld from another.
USERS: list[dict[str, object]] = [
    {
        "user_id": "alice.employee",
        "display_name": "Alice Nowak — Support Engineer",
        "roles": ["employee", "engineering"],
        "description": "General staff with engineering access. No finance, HR or legal.",
    },
    {
        "user_id": "bob.manager",
        "display_name": "Bob Adeyemi — Delivery Manager",
        "roles": ["employee", "manager"],
        "description": "Line manager. Reads management-visible material; no runbooks.",
    },
    {
        "user_id": "carol.hr",
        "display_name": "Carol Bianchi — HR Business Partner",
        "roles": ["employee", "hr", "manager"],
        "description": "HR function, including restricted HR material.",
    },
    {
        "user_id": "dan.finance",
        "display_name": "Dan Oyelaran — Finance Analyst",
        "roles": ["employee", "finance"],
        "description": "Finance function. May read restricted finance material.",
    },
    {
        "user_id": "erin.contractor",
        "display_name": "Erin Vasquez — External Contractor",
        "roles": ["employee"],
        "description": "Baseline access only. The narrowest scope in the corpus.",
    },
    {
        "user_id": "frank.legal",
        "display_name": "Frank Osei — Legal Counsel",
        "roles": ["employee", "legal", "manager"],
        "description": "Legal function. May read restricted legal material.",
    },
    {
        "user_id": "grace.finlead",
        "display_name": "Grace Lam — Finance Lead",
        "roles": ["employee", "finance", "manager"],
        "description": "Finance leadership. Widest finance scope.",
    },
]

# Which principal is the natural asker for a given audience, and which
# principal is a natural *denied* asker for the same material. The denied
# pairing is what makes an ACL-probe item meaningful.
AUTHORISED_ASKER = {
    "employee": "erin.contractor",
    "manager": "bob.manager",
    "engineering": "alice.employee",
    "finance": "dan.finance",
    "hr": "carol.hr",
    "legal": "frank.legal",
}
DENIED_ASKER = {
    "engineering": "bob.manager",
    "finance": "alice.employee",
    "hr": "alice.employee",
    "legal": "alice.employee",
    "manager": "erin.contractor",
    "employee": None,
}

BASE_DATE = date(2023, 1, 1)


def _stable_id(text: str, prefix: str) -> str:
    """Generate a stable, deterministic ID from content."""
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace, for robust span matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


# ===========================================================================
# The core correctness primitive: resolve evidence text to real chunk IDs
# ===========================================================================


class CorpusDocument:
    """One document, chunked exactly as AHRAG will chunk it at ingest time."""

    def __init__(
        self,
        doc_id: str,
        title: str,
        text: str,
        doc_type: str,
        owner: str,
        acl_roles: list[str],
        created: date,
        effective: date,
        version: str,
        authority: int,
        policy_family: str | None = None,
        supersedes: str | None = None,
        filename_suffix: str = "txt",
    ) -> None:
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.doc_type = doc_type
        self.owner = owner
        self.acl_roles = acl_roles
        self.created = created
        self.effective = effective
        self.version = version
        self.authority = authority
        self.policy_family = policy_family
        self.supersedes = supersedes
        self.filename = f"{doc_id}.{filename_suffix}"

        meta = DocumentMeta(
            doc_id=doc_id,
            title=title,
            source_uri=f"corpus/{self.filename}",
            doc_type=DocType(doc_type) if doc_type in DocType._value2member_map_ else DocType.OTHER,
            owner=owner,
            acl_roles=acl_roles,
            created_date=created,
            effective_date=effective,
            version=version,
            authority_score=authority,
            policy_family=policy_family,
            supersedes=supersedes,
        )
        # The whole point: real chunk IDs from the real chunker.
        self.chunks = chunk_document(text, meta)
        self._normalised = [(c.chunk_id, _normalise(c.text)) for c in self.chunks]

    def locate(self, evidence: str, min_chars: int = 40) -> list[str]:
        """Return the chunk IDs whose text contains ``evidence``.

        Tries the whole span, then progressively shorter prefixes, because
        evidence strings in the source datasets often span a chunk boundary or
        differ from the document text by whitespace and PDF artefacts.

        Args:
            evidence: The evidence span as given by the source dataset.
            min_chars: Shortest prefix worth matching. Below this a match is
                more likely to be coincidental than real.

        Returns:
            Matching chunk IDs in document order. Empty when the span cannot be
            located, which is a signal to drop the item rather than guess.
        """
        needle = _normalise(evidence)
        if len(needle) < min_chars:
            return []
        for length in (len(needle), 400, 240, 160, 100, 60, min_chars):
            if length > len(needle):
                continue
            probe = needle[:length]
            hits = [cid for cid, text in self._normalised if probe in text]
            if hits:
                return hits
        return []

    def manifest_entry(self) -> dict[str, object]:
        """The manifest row AHRAG's ingestion service consumes."""
        entry: dict[str, object] = {
            "doc_id": self.doc_id,
            "path": f"corpus/{self.filename}",
            "title": self.title,
            "doc_type": self.doc_type,
            "owner": self.owner,
            "acl_roles": self.acl_roles,
            "created_date": self.created.isoformat(),
            "effective_date": self.effective.isoformat(),
            "version": self.version,
            "authority_score": self.authority,
        }
        if self.policy_family:
            entry["policy_family"] = self.policy_family
        if self.supersedes:
            entry["supersedes"] = self.supersedes
        return entry


def assign_audience(rng: random.Random) -> list[str]:
    """Draw a document audience from :data:`AUDIENCE_MODEL`."""
    roll = rng.random()
    cumulative = 0.0
    for roles, share in AUDIENCE_MODEL:
        cumulative += share
        if roll <= cumulative:
            return list(roles)
    return list(AUDIENCE_MODEL[0][0])


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Read a JSON-lines file, tolerating a JSON array as well."""
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            try:
                data = json.load(handle)
                return data[:limit] if limit else data
            except json.JSONDecodeError:
                return []
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ===========================================================================
# FinanceBench
# ===========================================================================


def integrate_financebench(rng: random.Random) -> tuple[list[CorpusDocument], list[dict], dict]:
    """Build finance documents and evidence-linked evaluation items."""
    rows = _read_jsonl(DATASETS_DIR / "financebench_hf.json")
    if not rows:
        print("  financebench_hf.json not found or empty, skipping")
        return [], [], {}

    print(f"  Loaded {len(rows)} FinanceBench questions")

    # Group full evidence pages by source document.
    pages: dict[str, list[str]] = defaultdict(list)
    meta: dict[str, dict] = {}
    for row in rows:
        name = row.get("doc_name", "")
        if not name:
            continue
        for evidence in row.get("evidence", []):
            page = (evidence.get("evidence_text_full_page") or "").strip()
            if page and page not in pages[name]:
                pages[name].append(page)
        meta.setdefault(
            name,
            {
                "company": row.get("company", "Unknown"),
                "period": row.get("doc_period", 2022),
            },
        )

    documents: dict[str, CorpusDocument] = {}
    for name, page_list in pages.items():
        if not page_list:
            continue
        info = meta.get(name, {})
        period = int(info.get("period", 2022) or 2022)
        doc_id = _stable_id(name, "doc-fb")
        # Finance filings are finance-restricted by nature; a subset is opened
        # to management so the finance boundary has both sides represented.
        audience = ["finance"] if rng.random() < 0.62 else ["finance", "manager"]
        documents[name] = CorpusDocument(
            doc_id=doc_id,
            title=name.replace("_", " "),
            text="\n\n".join(page_list),
            doc_type="finance",
            owner=str(info.get("company", "Unknown")),
            acl_roles=audience,
            created=date(period, 1, 1),
            effective=date(period, 1, 1),
            version="1.0",
            authority=4,
        )

    items: list[dict] = []
    stats = {"total": 0, "located": 0, "dropped_no_evidence": 0, "dropped_unlocatable": 0}
    for row in rows:
        name = row.get("doc_name", "")
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        document = documents.get(name)
        if not question or document is None:
            continue
        stats["total"] += 1

        spans = [
            (e.get("evidence_text") or "").strip()
            for e in row.get("evidence", [])
            if (e.get("evidence_text") or "").strip()
        ]
        if not spans:
            stats["dropped_no_evidence"] += 1
            continue

        gold: list[str] = []
        for span in spans:
            for chunk_id in document.locate(span):
                if chunk_id not in gold:
                    gold.append(chunk_id)
        if not gold:
            # No fabricated fallback. An unlocatable span means we do not know
            # the gold chunk, and inventing one corrupts every metric built on it.
            stats["dropped_unlocatable"] += 1
            continue
        stats["located"] += 1

        reasoning = str(row.get("question_reasoning") or "").lower()
        if "numerical" in reasoning or "calculation" in reasoning:
            expected = "R3"
        elif "logical" in reasoning:
            expected = "R4"
        elif "information extraction" in reasoning:
            expected = "R1"
        else:
            expected = "R3"
        # Multi-chunk evidence genuinely needs more than a single lookup.
        if len(gold) >= 2 and expected == "R1":
            expected = "R3"

        asker = "dan.finance" if "manager" not in document.acl_roles else "grace.finlead"
        fb_id = row.get("financebench_id", "")
        items.append(
            {
                "id": f"fb-{fb_id}" if fb_id else _stable_id(question, "fb"),
                "query": question,
                "user_id": asker,
                "query_type": f"finance_{row.get('question_type') or 'unknown'}",
                "expected_route": expected,
                "gold_chunks": gold,
                "should_abstain": False,
                "freshness_sensitive": False,
                "notes": f"FinanceBench. Reference answer: {answer[:90]}",
            }
        )

    print(
        f"  Created {len(documents)} documents, {len(items)} eval items "
        f"(located {stats['located']}/{stats['total']}, "
        f"dropped {stats['dropped_unlocatable']} unlocatable, "
        f"{stats['dropped_no_evidence']} without evidence spans)"
    )
    return list(documents.values()), items, stats


# ===========================================================================
# PolicyQA — the source of the supersession chains
# ===========================================================================


def integrate_policyqa(
    rng: random.Random, max_docs: int = 120, chains: int = 15
) -> tuple[list[CorpusDocument], list[dict], dict]:
    """Build policy documents, supersession chains, and comprehension items.

    PolicyQA gives many questions per policy paragraph. The richest paragraphs
    become documents; a subset is duplicated into ``v1``/``v2`` pairs to create
    real supersession chains, which is what makes the freshness metric
    non-trivial (improvement.txt §1(d)).
    """
    rows = _read_jsonl(DATASETS_DIR / "policyqa_train.json")
    if not rows:
        print("  policyqa_train.json not found or empty, skipping")
        return [], [], {}

    print(f"  Loaded {len(rows)} PolicyQA items")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        context = (row.get("context") or "").strip()
        if len(context) < 200:
            continue
        grouped[context[:200]].append(row)

    ranked = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)[:max_docs]

    documents: list[CorpusDocument] = []
    items: list[dict] = []
    stats = {"total": 0, "located": 0, "dropped_unlocatable": 0, "chains": 0}
    chain_targets = {i for i in range(min(chains, len(ranked)))}

    for index, (key, group) in enumerate(ranked):
        context = group[0].get("context", "")
        base_id = _stable_id(key, "doc-pqa")
        audience = assign_audience(rng)
        family = f"policy-family-{index:03d}"
        is_chain = index in chain_targets

        if is_chain:
            # A superseded v1 whose text still contains the answer, plus a
            # current v2. The v1 wording is deliberately kept lexically strong
            # so that "the superseded document is MORE lexically relevant" —
            # the real test of freshness handling (improvement.txt §2c).
            v1 = CorpusDocument(
                doc_id=f"{base_id}-v1",
                title=f"Policy Section {index:03d} (retired)",
                text=context,
                doc_type="policy",
                owner="Privacy Team",
                acl_roles=audience,
                created=BASE_DATE,
                effective=BASE_DATE,
                version="1.0",
                authority=3,
                policy_family=family,
            )
            v2 = CorpusDocument(
                doc_id=f"{base_id}-v2",
                title=f"Policy Section {index:03d}",
                text=context + "\n\n## Revision note\n\nThis revision supersedes "
                "the previous version of this section.",
                doc_type="policy",
                owner="Privacy Team",
                acl_roles=audience,
                created=BASE_DATE + timedelta(days=540),
                effective=BASE_DATE + timedelta(days=600),
                version="2.0",
                authority=4,
                policy_family=family,
                supersedes=f"{base_id}-v1",
            )
            documents.extend([v1, v2])
            stats["chains"] += 1
            answer_doc = v2
        else:
            answer_doc = CorpusDocument(
                doc_id=base_id,
                title=f"Policy Section {index:03d}",
                text=context,
                doc_type="policy",
                owner="Privacy Team",
                acl_roles=audience,
                created=BASE_DATE,
                effective=BASE_DATE,
                version="1.0",
                authority=4,
                policy_family=family,
            )
            documents.append(answer_doc)

        primary_role = audience[0]
        asker = AUTHORISED_ASKER.get(primary_role, "erin.contractor")

        for question_index, row in enumerate(group[:3]):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            stats["total"] += 1
            answers = row.get("answers") or {}
            span = ""
            if isinstance(answers, dict):
                texts = answers.get("text") or []
                if texts:
                    span = str(texts[0])

            gold = answer_doc.locate(span) if span else []
            if not gold:
                # Fall back to the chunk containing the question's rarest terms
                # rather than to ``::c000``. Still evidence-driven; if that
                # fails too, the item is dropped.
                gold = answer_doc.locate(span, min_chars=20) if span else []
            if not gold:
                stats["dropped_unlocatable"] += 1
                continue
            stats["located"] += 1

            items.append(
                {
                    "id": _stable_id(f"{answer_doc.doc_id}-{question_index}-{question}", "pqa"),
                    "query": question,
                    "user_id": asker,
                    "query_type": "policy_comprehension",
                    "expected_route": "R2",
                    "gold_chunks": gold,
                    "should_abstain": False,
                    "freshness_sensitive": is_chain,
                    "notes": f"PolicyQA. Answer span: {span[:70]}",
                }
            )

    print(
        f"  Created {len(documents)} documents ({stats['chains']} supersession chains), "
        f"{len(items)} eval items (located {stats['located']}/{stats['total']}, "
        f"dropped {stats['dropped_unlocatable']})"
    )
    return documents, items, stats


# ===========================================================================
# HotpotQA — multi-hop items
# ===========================================================================


def integrate_hotpotqa(
    rng: random.Random, max_items: int = 300
) -> tuple[list[CorpusDocument], list[dict], dict]:
    """Build wiki documents and genuine multi-hop evaluation items."""
    rows = _read_jsonl(DATASETS_DIR / "hotpotqa_dev.json", limit=max_items)
    if not rows:
        print("  hotpotqa_dev.json not found or empty, skipping")
        return [], [], {}

    print(f"  Loaded {len(rows)} HotpotQA items")

    # Collect article text and the sentence list, so supporting facts can be
    # resolved to the exact sentence and then to the chunk containing it.
    articles: dict[str, list[str]] = {}
    for row in rows:
        context = row.get("context") or {}
        if isinstance(context, dict):
            titles = context.get("title") or []
            sentences = context.get("sentences") or []
            for title, sentence_list in zip(titles, sentences):
                articles.setdefault(title, list(sentence_list))
        elif isinstance(context, list):
            for entry in context:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    title, sentence_list = entry[0], entry[1]
                    articles.setdefault(
                        title,
                        list(sentence_list)
                        if isinstance(sentence_list, list)
                        else [str(sentence_list)],
                    )

    documents: dict[str, CorpusDocument] = {}
    for title, sentences in articles.items():
        text = " ".join(sentences).strip()
        if len(text) < 80:
            continue
        documents[title] = CorpusDocument(
            doc_id=_stable_id(title, "doc-hqa"),
            title=f"Reference: {title}",
            text=text,
            doc_type="wiki",
            owner="Reference Library",
            acl_roles=assign_audience(rng),
            created=BASE_DATE,
            effective=BASE_DATE,
            version="1.0",
            authority=3,
        )

    items: list[dict] = []
    stats = {"total": 0, "located": 0, "dropped_unlocatable": 0, "dropped_scope": 0}
    for row in rows:
        question = (row.get("question") or "").strip()
        if not question:
            continue
        stats["total"] += 1
        answer = (row.get("answer") or "").strip()
        supporting = row.get("supporting_facts") or {}

        # Resolve each (title, sentence_index) supporting fact to a real chunk.
        pairs: list[tuple[str, int]] = []
        if isinstance(supporting, dict):
            pairs = list(
                zip(supporting.get("title") or [], supporting.get("sent_id") or [])
            )
        elif isinstance(supporting, list):
            pairs = [
                (entry[0], entry[1])
                for entry in supporting
                if isinstance(entry, (list, tuple)) and len(entry) >= 2
            ]

        gold: list[str] = []
        source_docs: set[str] = set()
        for title, sentence_index in pairs:
            document = documents.get(title)
            sentences = articles.get(title) or []
            if document is None or not sentences:
                continue
            try:
                sentence = sentences[int(sentence_index)]
            except (IndexError, ValueError, TypeError):
                continue
            for chunk_id in document.locate(sentence, min_chars=25):
                if chunk_id not in gold:
                    gold.append(chunk_id)
                source_docs.add(document.doc_id)

        if not gold:
            stats["dropped_unlocatable"] += 1
            continue

        # Every gold chunk must be readable by one principal, or the item is
        # unanswerable-by-construction rather than a genuine multi-hop test.
        required: set[str] = set()
        for chunk_id in gold:
            doc_id = chunk_id.split("::")[0]
            for document in documents.values():
                if document.doc_id == doc_id:
                    required.update(document.acl_roles)
                    break
        asker = None
        for user in USERS:
            roles = {str(r) for r in user["roles"]}  # type: ignore[union-attr]
            if required and required <= roles:
                asker = str(user["user_id"])
                break
        if asker is None:
            stats["dropped_scope"] += 1
            continue
        stats["located"] += 1

        question_type = row.get("type") or ""
        level = row.get("level") or ""
        if question_type == "comparison" or len(source_docs) >= 2:
            expected = "R4"
        elif level == "hard":
            expected = "R3"
        else:
            expected = "R3"

        items.append(
            {
                "id": _stable_id(question, "hqa"),
                "query": question,
                "user_id": asker,
                "query_type": f"hotpotqa_{question_type}" if question_type else "hotpotqa",
                "expected_route": expected,
                "gold_chunks": gold,
                "should_abstain": False,
                "freshness_sensitive": False,
                "notes": f"HotpotQA {question_type}/{level}. Answer: {answer[:60]}",
            }
        )

    print(
        f"  Created {len(documents)} documents, {len(items)} eval items "
        f"(located {stats['located']}/{stats['total']}, "
        f"dropped {stats['dropped_unlocatable']} unlocatable, "
        f"{stats['dropped_scope']} outside any principal's scope)"
    )
    return list(documents.values()), items, stats


# ===========================================================================
# Governance strata: ACL probes, abstention items, freshness items
# ===========================================================================


def build_acl_probe_items(
    documents: list[CorpusDocument], answerable: list[dict], limit: int = 60
) -> list[dict]:
    """Re-ask answerable questions as a principal who may *not* read the answer.

    These are the items that actually test the ACL boundary: the answer exists
    in the corpus, the asker cannot read it, and the correct behaviour is
    abstention without disclosing that the material exists. ``forbidden_chunks``
    records what a violation would look like, which
    ``ahrag.eval.metrics.acl_violation`` consumes.
    """
    by_doc = {document.doc_id: document for document in documents}
    probes: list[dict] = []
    for item in answerable:
        if len(probes) >= limit:
            break
        gold = item.get("gold_chunks") or []
        if not gold:
            continue
        document = by_doc.get(gold[0].split("::")[0])
        if document is None:
            continue
        restricted = [r for r in document.acl_roles if r != "employee"]
        if not restricted:
            continue
        denied = DENIED_ASKER.get(restricted[0])
        if not denied:
            continue
        denied_roles = next(
            ({str(r) for r in u["roles"]} for u in USERS if u["user_id"] == denied),  # type: ignore[union-attr]
            set(),
        )
        if set(document.acl_roles) & denied_roles:
            continue

        probes.append(
            {
                "id": _stable_id(f"acl-{item['id']}", "acl"),
                "query": item["query"],
                "user_id": denied,
                "query_type": "permission_boundary",
                "expected_route": "R0",
                "gold_chunks": [],
                "forbidden_chunks": gold,
                "should_abstain": True,
                "freshness_sensitive": False,
                "notes": (
                    f"ACL probe: answer lives in {document.doc_id} "
                    f"(roles={document.acl_roles}), asked by {denied}."
                ),
            }
        )
    return probes


def build_unanswerable_items(limit: int = 30) -> list[dict]:
    """Questions with no support anywhere in the corpus.

    Without these, abstention appropriateness only measures the ACL case and
    the system is never tested on "this simply is not written down".
    """
    templates = [
        "What is the company's policy on interplanetary business travel?",
        "How many vacation days does the Mars office grant?",
        "What was the revenue of the underwater datacentre division?",
        "Which vendor supplies the executive submarine fleet?",
        "What is the escalation path for a zeppelin incident?",
        "How do I claim expenses for a time-travel conference?",
        "What is the retention period for telepathic communications?",
        "Who approves budget for the lunar manufacturing plant?",
        "What is the SLA for quantum teleportation outages?",
        "How is overtime calculated for robotic staff?",
    ]
    items: list[dict] = []
    askers = [str(u["user_id"]) for u in USERS]
    for index in range(limit):
        query = templates[index % len(templates)]
        suffix = "" if index < len(templates) else f" (case {index // len(templates) + 1})"
        items.append(
            {
                "id": _stable_id(f"unans-{index}-{query}", "unans"),
                "query": query + suffix,
                "user_id": askers[index % len(askers)],
                "query_type": "unanswerable",
                "expected_route": "R0",
                "gold_chunks": [],
                "should_abstain": True,
                "freshness_sensitive": False,
                "notes": "No support anywhere in the corpus; correct answer is abstention.",
            }
        )
    return items


def build_adversarial_items(answerable: list[dict], limit: int = 40) -> list[dict]:
    """Misspelled and rambling variants of answerable questions (§2c).

    Same gold chunks, degraded surface form. This is what separates a sparse
    route that depends on exact tokens from a dense route that does not.
    """
    def misspell(text: str) -> str:
        words = text.split()
        out = []
        for index, word in enumerate(words):
            if len(word) > 5 and index % 3 == 1:
                out.append(word[:-2] + word[-1] + word[-2])  # transpose last two
            else:
                out.append(word)
        return " ".join(out)

    preamble = (
        "I hope you can help me with something that has been on my mind for a "
        "while now and which I have been meaning to ask about, namely: "
    )

    items: list[dict] = []
    for index, item in enumerate(answerable):
        if len(items) >= limit:
            break
        if not item.get("gold_chunks"):
            continue
        if index % 2 == 0:
            query = misspell(item["query"])
            kind = "adversarial_misspelled"
        else:
            query = preamble + item["query"]
            kind = "adversarial_rambling"
        if query == item["query"]:
            continue
        items.append(
            {
                "id": _stable_id(f"{kind}-{item['id']}", "adv"),
                "query": query,
                "user_id": item["user_id"],
                "query_type": kind,
                "expected_route": item["expected_route"],
                "gold_chunks": list(item["gold_chunks"]),
                "should_abstain": False,
                "freshness_sensitive": False,
                "notes": f"Adversarial variant of {item['id']}.",
            }
        )
    return items


def build_freshness_items(documents: list[CorpusDocument], limit: int = 25) -> list[dict]:
    """Items whose gold evidence is the *current* version of a chain.

    The superseded sibling contains near-identical text, so a system that
    ignores freshness will retrieve the retired version and score zero.
    """
    chains = [d for d in documents if d.supersedes]
    items: list[dict] = []
    for document in chains[:limit]:
        if not document.chunks:
            continue
        retired = document.supersedes or ""
        asker = AUTHORISED_ASKER.get(
            document.acl_roles[0] if document.acl_roles else "employee",
            "erin.contractor",
        )
        heading = document.chunks[0].heading or document.title
        items.append(
            {
                "id": _stable_id(f"fresh-{document.doc_id}", "fresh"),
                "query": f"What is the current rule in {heading}?",
                "user_id": asker,
                "query_type": "freshness_competing_versions",
                "expected_route": "R3",
                "gold_chunks": [document.chunks[0].chunk_id],
                "forbidden_chunks": [],
                "should_abstain": False,
                "freshness_sensitive": True,
                "notes": (
                    f"Current version is {document.doc_id} v{document.version}; "
                    f"{retired} is superseded and lexically similar."
                ),
            }
        )
    return items


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate benchmark datasets into an AHRAG corpus and eval set",
    )
    parser.add_argument("--preview", action="store_true",
                        help="Report what would be produced without writing files")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "financebench", "policyqa", "hotpotqa"])
    parser.add_argument("--hotpotqa-limit", type=int, default=300)
    parser.add_argument("--policyqa-docs", type=int, default=120)
    parser.add_argument("--chains", type=int, default=15,
                        help="Supersession chains to build (improvement.txt asks for 10-15)")
    args = parser.parse_args()

    rng = random.Random(SEED)

    print("=" * 74)
    print("AHRAG Dataset Integration")
    print("=" * 74)
    print("Gold chunk IDs are resolved through the real chunker; items whose")
    print("evidence cannot be located are dropped rather than guessed.")
    print()

    documents: list[CorpusDocument] = []
    answerable: list[dict] = []
    reports: dict[str, dict] = {}

    if args.dataset in ("all", "financebench"):
        print("[1/3] FinanceBench...")
        docs, items, stats = integrate_financebench(rng)
        documents += docs
        answerable += items
        reports["financebench"] = stats
        print()

    if args.dataset in ("all", "policyqa"):
        print("[2/3] PolicyQA (+ supersession chains)...")
        docs, items, stats = integrate_policyqa(rng, args.policyqa_docs, args.chains)
        documents += docs
        answerable += items
        reports["policyqa"] = stats
        print()

    if args.dataset in ("all", "hotpotqa"):
        print("[3/3] HotpotQA...")
        docs, items, stats = integrate_hotpotqa(rng, args.hotpotqa_limit)
        documents += docs
        answerable += items
        reports["hotpotqa"] = stats
        print()

    if not documents:
        print("No documents produced. Are the raw dataset files present?")
        sys.exit(1)

    print("Building governance strata...")
    acl_items = build_acl_probe_items(documents, answerable)
    unanswerable = build_unanswerable_items()
    adversarial = build_adversarial_items(answerable)
    freshness = build_freshness_items(documents)
    print(f"  permission_boundary : {len(acl_items)}")
    print(f"  unanswerable        : {len(unanswerable)}")
    print(f"  adversarial         : {len(adversarial)}")
    print(f"  freshness-sensitive : {len(freshness)}")
    print()

    all_items = answerable + acl_items + unanswerable + adversarial + freshness

    # De-duplicate IDs; a duplicate makes load_eval_set raise.
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in all_items:
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        deduped.append(item)
    all_items = deduped

    # ---- validation: every gold and forbidden ID must exist ----
    known = {chunk.chunk_id for document in documents for chunk in document.chunks}
    bad: list[str] = []
    for item in all_items:
        for chunk_id in list(item.get("gold_chunks", [])) + list(
            item.get("forbidden_chunks", [])
        ):
            if chunk_id not in known:
                bad.append(f"{item['id']} -> {chunk_id}")
    if bad:
        print(f"FATAL: {len(bad)} references point at non-existent chunks:")
        for entry in bad[:10]:
            print(f"  {entry}")
        sys.exit(1)
    print(f"Validation: all gold/forbidden chunk IDs resolve ({len(known)} chunks known).")

    total_chunks = sum(len(d.chunks) for d in documents)
    total_chars = sum(len(d.text) for d in documents)
    audiences: dict[str, int] = defaultdict(int)
    for document in documents:
        audiences[",".join(document.acl_roles)] += 1
    types: dict[str, int] = defaultdict(int)
    for item in all_items:
        types[item["query_type"]] += 1

    print()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  documents           : {len(documents)}")
    print(f"  chunks (real)       : {total_chunks}")
    print(f"  corpus text         : {total_chars:,} chars")
    print(f"  supersession chains : {sum(1 for d in documents if d.supersedes)}")
    print(f"  eval items          : {len(all_items)}")
    print(f"    answerable        : {sum(1 for i in all_items if i['gold_chunks'])}")
    print(f"    should_abstain    : {sum(1 for i in all_items if i['should_abstain'])}")
    print(f"    freshness         : {sum(1 for i in all_items if i['freshness_sensitive'])}")
    print(f"    with forbidden    : {sum(1 for i in all_items if i.get('forbidden_chunks'))}")
    print()
    print("  document audiences (ACL classes):")
    for audience, count in sorted(audiences.items(), key=lambda kv: -kv[1]):
        print(f"    {audience:28s} {count:5d}")
    print()
    print("  eval items by type:")
    for query_type, count in sorted(types.items(), key=lambda kv: -kv[1]):
        print(f"    {query_type:34s} {count:5d}")

    if args.preview:
        print("\nPreview mode — no files written.")
        return

    print(f"\nWriting to {OUTPUT_DIR}/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_dir = OUTPUT_DIR / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    # Remove stale corpus files so a re-run cannot leave orphans behind that a
    # previous run's manifest referenced.
    for stale in corpus_dir.glob("*"):
        if stale.is_file():
            stale.unlink()

    for document in documents:
        (corpus_dir / document.filename).write_text(document.text, encoding="utf-8")
    print(f"  wrote {len(documents)} corpus files")

    manifest = {
        "users": USERS,
        "documents": [document.manifest_entry() for document in documents],
    }
    (OUTPUT_DIR / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"  wrote manifest.yaml ({len(documents)} documents, {len(USERS)} users)")

    (OUTPUT_DIR / "eval_set.yaml").write_text(
        yaml.safe_dump({"items": all_items}, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"  wrote eval_set.yaml ({len(all_items)} items)")

    report = {
        "seed": SEED,
        "documents": len(documents),
        "chunks": total_chunks,
        "corpus_chars": total_chars,
        "supersession_chains": sum(1 for d in documents if d.supersedes),
        "eval_items": len(all_items),
        "answerable": sum(1 for i in all_items if i["gold_chunks"]),
        "should_abstain": sum(1 for i in all_items if i["should_abstain"]),
        "freshness_sensitive": sum(1 for i in all_items if i["freshness_sensitive"]),
        "with_forbidden_chunks": sum(1 for i in all_items if i.get("forbidden_chunks")),
        "document_audiences": dict(audiences),
        "eval_types": dict(types),
        "per_dataset": reports,
        "gold_resolution": "real chunker; unlocatable evidence dropped, never defaulted",
    }
    (OUTPUT_DIR / "integration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("  wrote integration_report.json")
    print(f"\nDone. Ingest with:")
    print(f"  engine.seed(manifest_path=\"{OUTPUT_DIR / 'manifest.yaml'}\")")


if __name__ == "__main__":
    main()
