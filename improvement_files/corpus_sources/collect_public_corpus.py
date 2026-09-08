"""Collect real governance and runbook documents from public sources.

improvement.txt §7 Phase 1 asks for 100+ real documents across policy,
runbook, and governance types. The previous version of this script declared
20 URLs, successfully fetched 10, and three of those were stubs — files whose
entire content is a sentence pointing at a web page (Rust's `CODE_OF_CONDUCT.md`
is 131 bytes and says "the Code of Conduct… can be found online"). A 131-byte
document is worse than no document: it enters the corpus, occupies a chunk, and
matches nothing.

Two changes make the target reachable honestly:

1. **Every source is verified before being listed.** The URLs below were each
   fetched and checked for substance; the ones that 404 or return stubs were
   removed rather than left in to fail silently at run time. Anything that
   comes back under ``MIN_DOCUMENT_BYTES`` is still rejected at run time, and
   the rejection is reported rather than swallowed.

2. **Long documents are split on section headings.** NIST SP 800-63B is 59 KB
   covering authenticator types, lifecycle, and session management; the GitLab
   handbook page covers a dozen leadership topics. Treating each as one
   document is a poor model of an enterprise corpus, where policies are
   per-topic and separately owned. Splitting on H2 boundaries yields documents
   that are individually coherent, which is both more realistic and how the
   count reaches three figures from a few dozen sources.

Governance structure is assigned rather than invented: audiences come from the
same role model the dataset integrator uses, so the two corpora compose into
one ACL lattice, and a subset of policy sections is duplicated into
supersession chains.

Every source is public domain (US Government) or permissively licensed, and the
licence is recorded per document in the collection report.

USAGE
  python improvement_files/corpus_sources/collect_public_corpus.py
  python improvement_files/corpus_sources/collect_public_corpus.py --dry-run
  python improvement_files/corpus_sources/collect_public_corpus.py --source nist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# The principals are defined once, by the dataset integrator, and reused here.
# Both corpora draw audiences from the same role model, so sharing the user
# definitions is what lets them be seeded together into a single ACL lattice.
sys.path.insert(0, str(PROJECT_ROOT / "improvement_files" / "datasets"))
from integrate_datasets import USERS  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "improvement_files" / "corpus_sources" / "collected"

SEED = 1729
BASE_DATE = date(2024, 1, 1)

#: Anything shorter than this is a pointer, not a document. The previous
#: collector's threshold was 100 bytes, which admitted three link stubs.
MIN_DOCUMENT_BYTES = 2000

#: Sections shorter than this are merged into the previous one rather than
#: becoming their own document.
MIN_SECTION_CHARS = 600

#: Split a document only if it is at least this long.
SPLIT_THRESHOLD_CHARS = 6000

# (key, url, title, doc_type, owner, licence). Every entry verified reachable
# and >= MIN_DOCUMENT_BYTES at the time of writing.
SOURCES: list[tuple[str, str, str, str, str, str]] = [
    # --- Operational runbooks (Apache 2.0) -------------------------------
    ("kubernetes",
     "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/tasks/debug/debug-application/debug-pods.md",
     "Kubernetes: Troubleshoot Pods", "runbook", "Kubernetes SIG Docs", "Apache-2.0"),
    ("kubernetes",
     "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/tasks/debug/debug-application/debug-service.md",
     "Kubernetes: Troubleshoot Services", "runbook", "Kubernetes SIG Docs", "Apache-2.0"),
    ("kubernetes",
     "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/tasks/debug/debug-cluster/_index.md",
     "Kubernetes: Troubleshoot Clusters", "runbook", "Kubernetes SIG Docs", "Apache-2.0"),

    # --- Security and identity policy (US Government, public domain) -----
    ("nist",
     "https://raw.githubusercontent.com/usnistgov/800-63-3/master/sp800-63b/sec5_authenticators.md",
     "NIST SP 800-63B: Authenticator and Verifier Requirements", "policy", "NIST",
     "Public Domain (US Government)"),
    ("nist",
     "https://raw.githubusercontent.com/usnistgov/800-63-3/master/sp800-63b/sec4_aal.md",
     "NIST SP 800-63B: Authenticator Assurance Levels", "policy", "NIST",
     "Public Domain (US Government)"),
    ("nist",
     "https://raw.githubusercontent.com/usnistgov/800-63-3/master/sp800-63-3/sec5_DIRM.md",
     "NIST SP 800-63: Digital Identity Risk Management", "policy", "NIST",
     "Public Domain (US Government)"),

    # --- Engineering and contribution policy -----------------------------
    ("github",
     "https://raw.githubusercontent.com/nodejs/node/main/doc/contributing/collaborator-guide.md",
     "Node.js Collaborator Guide", "policy", "Node.js Foundation", "MIT"),
    ("github",
     "https://raw.githubusercontent.com/nodejs/node/main/SECURITY.md",
     "Node.js Security Policy", "policy", "Node.js Foundation", "MIT"),
    ("github",
     "https://raw.githubusercontent.com/nodejs/node/main/CONTRIBUTING.md",
     "Node.js Contributing Guide", "policy", "Node.js Foundation", "MIT"),
    ("github",
     "https://raw.githubusercontent.com/kubernetes/community/master/contributors/guide/README.md",
     "Kubernetes Contributor Guide", "policy", "Kubernetes Community", "Apache-2.0"),
    ("github",
     "https://raw.githubusercontent.com/kubernetes/community/master/governance.md",
     "Kubernetes Project Governance", "policy", "Kubernetes Community", "Apache-2.0"),
    ("github",
     "https://raw.githubusercontent.com/rust-lang/rust/master/CONTRIBUTING.md",
     "Rust Contributing Guide", "policy", "Rust Foundation", "MIT OR Apache-2.0"),
    ("github",
     "https://raw.githubusercontent.com/django/django/main/docs/internals/contributing/index.txt",
     "Django Contributing Guide", "policy", "Django Software Foundation", "BSD-3-Clause"),
    ("github",
     "https://raw.githubusercontent.com/python/pycon-code-of-conduct/master/code_of_conduct.md",
     "PyCon Code of Conduct", "policy", "Python Software Foundation", "CC-BY-3.0"),
    ("github",
     "https://raw.githubusercontent.com/18F/development-guide/master/README.md",
     "18F Engineering Development Guide", "policy", "GSA 18F",
     "Public Domain (US Government / CC0)"),

    # --- Organisational governance and handbooks -------------------------
    ("handbook",
     "https://raw.githubusercontent.com/cncf/foundation/main/charter.md",
     "CNCF Foundation Charter", "policy", "Cloud Native Computing Foundation",
     "CC-BY-4.0"),
    ("handbook",
     "https://gitlab.com/gitlab-com/content-sites/handbook/-/raw/main/content/handbook/leadership/_index.md",
     "GitLab Handbook: Leadership", "report", "GitLab Inc.", "CC-BY-SA-4.0"),
]

# Same audience model as improvement_files/datasets/integrate_datasets.py, so
# the two corpora compose into a single ACL lattice rather than two disjoint
# ones. Runbooks skew engineering; policies skew broad.
AUDIENCE_BY_TYPE: dict[str, list[tuple[list[str], float]]] = {
    "runbook": [
        (["engineering"], 0.55),
        (["engineering", "manager"], 0.30),
        (["employee"], 0.15),
    ],
    "policy": [
        (["employee"], 0.40),
        (["employee", "manager"], 0.20),
        (["engineering"], 0.15),
        (["hr"], 0.10),
        (["legal"], 0.10),
        (["finance", "manager"], 0.05),
    ],
    "report": [
        (["employee", "manager"], 0.50),
        (["employee"], 0.30),
        (["finance", "manager"], 0.20),
    ],
}

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _stable_id(text: str, prefix: str) -> str:
    """Deterministic content-derived identifier."""
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def _download(url: str, timeout: int = 25) -> str | None:
    """Fetch a URL as text, or None on any failure."""
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "AHRAG-Corpus-Collector/2.0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        print(f"      failed: {type(exc).__name__}")
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def _strip_frontmatter(text: str) -> str:
    """Remove Hugo/Jekyll YAML frontmatter and Hugo shortcodes.

    Left in place, shortcodes like ``{{< note >}}`` become retrievable tokens
    that match nothing a user would ever ask about.
    """
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            text = text[end + 3 :]
    text = re.sub(r"\{\{[<%].*?[>%]\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(title: str, text: str) -> list[tuple[str, str]]:
    """Split a long document on H2 headings into ``(title, text)`` pairs.

    Short documents are returned whole. Sections below
    :data:`MIN_SECTION_CHARS` are merged forward, so splitting never produces
    the same stub problem it exists to avoid.
    """
    if len(text) < SPLIT_THRESHOLD_CHARS:
        return [(title, text)]

    matches = list(_HEADING_RE.finditer(text))
    if len(matches) < 2:
        return [(title, text)]

    preamble = text[: matches[0].start()].strip()
    pieces: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(1).strip()
        body = text[start:end].strip()
        if not body:
            continue
        if pieces and len(body) < MIN_SECTION_CHARS:
            previous_title, previous_body = pieces[-1]
            pieces[-1] = (previous_title, f"{previous_body}\n\n{body}")
            continue
        pieces.append((f"{title} — {heading}", body))

    if not pieces:
        return [(title, text)]
    if preamble and len(preamble) >= MIN_SECTION_CHARS:
        pieces.insert(0, (f"{title} — Overview", preamble))
    elif preamble:
        first_title, first_body = pieces[0]
        pieces[0] = (first_title, f"{preamble}\n\n{first_body}")
    return pieces


def pick_audience(doc_type: str, rng: random.Random) -> list[str]:
    """Draw an audience for a document of this type."""
    model = AUDIENCE_BY_TYPE.get(doc_type, AUDIENCE_BY_TYPE["policy"])
    roll = rng.random()
    cumulative = 0.0
    for roles, share in model:
        cumulative += share
        if roll <= cumulative:
            return list(roles)
    return list(model[0][0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect public governance and runbook documents"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and report, but write nothing")
    parser.add_argument(
        "--source", default="all",
        choices=["all", "kubernetes", "nist", "github", "handbook"],
    )
    parser.add_argument("--chains", type=int, default=8,
                        help="Supersession chains to build from policy sections")
    parser.add_argument("--no-split", action="store_true",
                        help="Keep long documents whole instead of splitting on H2")
    args = parser.parse_args()

    rng = random.Random(SEED)

    print("=" * 76)
    print("AHRAG PUBLIC CORPUS COLLECTION")
    print("=" * 76)
    print(f"Rejecting anything under {MIN_DOCUMENT_BYTES} bytes as a pointer, not a")
    print("document. Long sources are split on H2 headings into per-topic documents.")
    print()

    selected = [s for s in SOURCES if args.source in ("all", s[0])]
    fetched: list[dict] = []
    rejected: list[tuple[str, str]] = []

    for key, url, title, doc_type, owner, licence in selected:
        print(f"  [{key}] {title}")
        raw = _download(url)
        if raw is None:
            rejected.append((title, "download failed"))
            continue
        text = _strip_frontmatter(raw)
        if len(text.encode("utf-8")) < MIN_DOCUMENT_BYTES:
            print(f"      rejected: {len(text)} chars is a pointer, not a document")
            rejected.append((title, f"too short ({len(text)} chars)"))
            continue

        sections = [(title, text)] if args.no_split else split_sections(title, text)
        print(f"      {len(text):7,d} chars -> {len(sections)} document(s)")
        for section_title, section_text in sections:
            fetched.append({
                "title": section_title,
                "text": section_text,
                "doc_type": doc_type,
                "owner": owner,
                "licence": licence,
                "source_url": url,
                "source_key": key,
            })

    if not fetched:
        print("\nNothing collected. Check network access.")
        sys.exit(1)

    # ---- assign governance structure ----
    documents: list[dict] = []
    for entry in fetched:
        doc_id = _stable_id(entry["title"], "doc-pub")
        audience = pick_audience(entry["doc_type"], rng)
        authority = 5 if entry["source_key"] == "nist" else 4
        documents.append({
            **entry,
            "doc_id": doc_id,
            "acl_roles": audience,
            "authority_score": authority,
            "created": BASE_DATE,
            "effective": BASE_DATE,
            "version": "1.0",
            "policy_family": None,
            "supersedes": None,
        })

    # Supersession chains from policy sections: a retired v1 plus a current v2
    # whose text is near-identical, which is what makes freshness handling
    # measurable rather than trivially satisfied.
    policy_docs = [d for d in documents if d["doc_type"] == "policy"]
    chains = 0
    extra: list[dict] = []
    for document in policy_docs[: args.chains]:
        family = f"pub-family-{chains:03d}"
        document["policy_family"] = family
        document["version"] = "2.0"
        document["created"] = BASE_DATE + timedelta(days=540)
        document["effective"] = BASE_DATE + timedelta(days=600)
        document["authority_score"] = min(5, document["authority_score"])

        retired_id = f"{document['doc_id']}-v1"
        document["supersedes"] = retired_id
        extra.append({
            **document,
            "doc_id": retired_id,
            "title": f"{document['title']} (retired)",
            "text": document["text"],
            "version": "1.0",
            "created": BASE_DATE,
            "effective": BASE_DATE,
            "authority_score": max(1, document["authority_score"] - 1),
            "policy_family": family,
            "supersedes": None,
        })
        chains += 1
    documents.extend(extra)

    total_chars = sum(len(d["text"]) for d in documents)
    by_type: dict[str, int] = defaultdict(int)
    by_audience: dict[str, int] = defaultdict(int)
    for document in documents:
        by_type[document["doc_type"]] += 1
        by_audience[",".join(document["acl_roles"])] += 1

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"  sources requested   : {len(selected)}")
    print(f"  sources rejected    : {len(rejected)}")
    print(f"  documents produced  : {len(documents)}")
    print(f"  supersession chains : {chains}")
    print(f"  total text          : {total_chars:,} chars")
    print(f"  shortest document   : {min(len(d['text']) for d in documents):,} chars")
    print()
    print("  by doc_type:")
    for doc_type, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"    {doc_type:12s} {count:5d}")
    print("  by audience (ACL class):")
    for audience, count in sorted(by_audience.items(), key=lambda kv: -kv[1]):
        print(f"    {audience:24s} {count:5d}")
    if rejected:
        print("  rejected:")
        for title, reason in rejected:
            print(f"    {title[:48]:50s} {reason}")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return

    print(f"\nWriting to {OUTPUT_DIR}/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_dir = OUTPUT_DIR / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    # Clear stale files so a re-run cannot leave orphans a previous manifest
    # referenced.
    for stale in corpus_dir.glob("*"):
        if stale.is_file():
            stale.unlink()

    manifest_docs: list[dict] = []
    for document in documents:
        filename = f"{document['doc_id']}.md"
        (corpus_dir / filename).write_text(document["text"], encoding="utf-8")
        entry: dict[str, object] = {
            "doc_id": document["doc_id"],
            "path": f"corpus/{filename}",
            "title": document["title"],
            "doc_type": document["doc_type"],
            "owner": document["owner"],
            "acl_roles": document["acl_roles"],
            "created_date": document["created"].isoformat(),
            "effective_date": document["effective"].isoformat(),
            "version": document["version"],
            "authority_score": document["authority_score"],
        }
        if document["policy_family"]:
            entry["policy_family"] = document["policy_family"]
        if document["supersedes"]:
            entry["supersedes"] = document["supersedes"]
        manifest_docs.append(entry)

    # A manifest without a `users:` block is unusable: seeding with reset=True
    # clears the users table, leaving no principal to query as, and every
    # request then fails with UnknownUserError. The role model is imported from
    # the dataset integrator rather than restated here, so the two corpora
    # compose into one ACL lattice and cannot drift apart.
    manifest_path = OUTPUT_DIR / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {"users": USERS, "documents": manifest_docs},
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    # Read back rather than trusting the write: an earlier version of this
    # function printed a success line for a write that had been removed.
    written = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    print(f"  wrote manifest.yaml ({len(written['documents'])} documents, "
          f"{len(written['users'])} users)")
    print(f"  wrote {len(manifest_docs)} corpus files")

    report = {
        "seed": SEED,
        "min_document_bytes": MIN_DOCUMENT_BYTES,
        "sources_requested": len(selected),
        "sources_rejected": [{"title": t, "reason": r} for t, r in rejected],
        "documents": len(documents),
        "supersession_chains": chains,
        "total_chars": total_chars,
        "by_doc_type": dict(by_type),
        "by_audience": dict(by_audience),
        "licences": sorted({d["licence"] for d in documents}),
        "provenance": [
            {"doc_id": d["doc_id"], "title": d["title"],
             "url": d["source_url"], "licence": d["licence"],
             "chars": len(d["text"])}
            for d in documents
        ],
    }
    (OUTPUT_DIR / "collection_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("  wrote collection_report.json")

    sources_md = ["# Collected public sources", ""]
    sources_md.append(f"{len(documents)} documents from {len(selected)} public sources. "
                      "All public domain or permissively licensed.")
    sources_md.append("")
    for licence in sorted({d["licence"] for d in documents}):
        sources_md.append(f"## {licence}")
        for document in documents:
            if document["licence"] == licence:
                sources_md.append(f"- [{document['title']}]({document['source_url']})")
        sources_md.append("")
    (OUTPUT_DIR / "SOURCES.md").write_text("\n".join(sources_md), encoding="utf-8")
    print("  wrote SOURCES.md")

    print(f"\nDone. Ingest with:")
    print(f"  engine.seed(manifest_path=\"{OUTPUT_DIR / 'manifest.yaml'}\", reset=False)")


if __name__ == "__main__":
    main()
