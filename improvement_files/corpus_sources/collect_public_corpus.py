"""
Public Corpus Collection for AHRAG
=====================================
Downloads freely-available documents from public sources to expand AHRAG's
corpus beyond the 9-document seed.

ALL sources are:
  - Public domain (US Government) or permissively licensed (Apache 2.0, MIT, CC)
  - Real documents with substantive content (not lorem ipsum or templates)
  - Relevant to AHRAG's governance-aware RAG thesis (policies, runbooks, reports)

SOURCES:
  1. US OPM (Office of Personnel Management) policy guidance — public domain
  2. Kubernetes troubleshooting docs — Apache 2.0
  3. NIST Cybersecurity publications — public domain
  4. Open-source project contribution guidelines from GitHub — MIT/Apache 2.0

OUTPUT:
  improvement_files/corpus_sources/collected/
    ├── manifest.yaml           (AHRAG-compatible manifest)
    ├── corpus/                 (downloaded document text files)
    └── collection_report.json  (metadata and statistics)

USAGE:
  python improvement_files/corpus_sources/collect_public_corpus.py
  python improvement_files/corpus_sources/collect_public_corpus.py --dry-run
  python improvement_files/corpus_sources/collect_public_corpus.py --source opm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "improvement_files" / "corpus_sources" / "collected"


def _stable_id(text: str, prefix: str) -> str:
    """Generate a stable deterministic ID."""
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def _download(url: str, timeout: int = 30) -> str | None:
    """Download a URL and return the text content, or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AHRAG-Corpus-Collector/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            # Try UTF-8 first, fall back to latin-1
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        print(f"    Failed: {exc}")
        return None


def _strip_html_basic(html: str) -> str:
    """Very basic HTML tag stripping for converting HTML to readable text."""
    import re
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Replace block elements with newlines
    text = re.sub(r"<(?:p|div|h[1-6]|li|br|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Clean up whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r" +", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return text.strip()


# ============================================================================
# Source 1: Kubernetes Troubleshooting Documentation (Apache 2.0)
# ============================================================================

def collect_kubernetes_docs() -> list[dict]:
    """
    Download Kubernetes troubleshooting documentation from the official
    GitHub repository (Apache 2.0 licensed).

    These are real operational runbooks — exactly the kind of document
    AHRAG's governance-aware routing is designed to handle.
    """
    # Raw markdown files from the official Kubernetes website repo
    base_url = "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/tasks/debug/"

    pages = [
        ("debug-application/debug-pods/index.md", "Troubleshoot Pods", "runbook"),
        ("debug-application/debug-service/index.md", "Troubleshoot Services", "runbook"),
        ("debug-application/debug-statefulset/index.md", "Debug StatefulSets", "runbook"),
        ("debug-application/debug-init-containers/index.md", "Debug Init Containers", "runbook"),
        ("debug-cluster/index.md", "Troubleshoot Clusters", "runbook"),
        ("debug-application/debug-running-pod/index.md", "Debug Running Pods", "runbook"),
        ("debug-application/get-shell-running-container/index.md", "Get Shell into Container", "runbook"),
        ("debug-application/determine-reason-pod-failure/index.md", "Determine Pod Failure Reason", "runbook"),
    ]

    documents = []
    for path, title, doc_type in pages:
        url = base_url + path
        print(f"    Downloading: {title}...")
        content = _download(url)
        if not content or len(content) < 200:
            print(f"    Skipped (too short or failed)")
            continue

        # Strip Hugo frontmatter
        if content.startswith("---"):
            end = content.find("---", 3)
            if end > 0:
                content = content[end + 3:].strip()

        documents.append({
            "title": f"Kubernetes: {title}",
            "text": content,
            "doc_type": doc_type,
            "owner": "Kubernetes SIG Docs",
            "acl_roles": ["engineering", "employee"],
            "authority_score": 4,
            "source_url": url,
            "license": "Apache 2.0",
        })

    print(f"    Collected {len(documents)} Kubernetes docs")
    return documents


# ============================================================================
# Source 2: GitHub Open Source Policy Documents (MIT/Apache 2.0)
# ============================================================================

def collect_github_policies() -> list[dict]:
    """
    Download real policy and governance documents from well-known open source
    projects. These are actual contribution guidelines, codes of conduct,
    and security policies — governance documents by nature.
    """
    # Format: (raw URL, title, doc_type, owner, license)
    sources = [
        (
            "https://raw.githubusercontent.com/nodejs/node/main/CODE_OF_CONDUCT.md",
            "Node.js Code of Conduct", "policy", "Node.js Foundation", "MIT",
        ),
        (
            "https://raw.githubusercontent.com/nodejs/node/main/CONTRIBUTING.md",
            "Node.js Contributing Guide", "policy", "Node.js Foundation", "MIT",
        ),
        (
            "https://raw.githubusercontent.com/nodejs/node/main/SECURITY.md",
            "Node.js Security Policy", "policy", "Node.js Foundation", "MIT",
        ),
        (
            "https://raw.githubusercontent.com/rust-lang/rust/master/CODE_OF_CONDUCT.md",
            "Rust Code of Conduct", "policy", "Rust Foundation", "MIT/Apache 2.0",
        ),
        (
            "https://raw.githubusercontent.com/rust-lang/rust/master/CONTRIBUTING.md",
            "Rust Contributing Guide", "policy", "Rust Foundation", "MIT/Apache 2.0",
        ),
        (
            "https://raw.githubusercontent.com/python/cpython/main/CODE_OF_CONDUCT.md",
            "Python Code of Conduct", "policy", "Python Software Foundation", "PSF-2.0",
        ),
        (
            "https://raw.githubusercontent.com/python/cpython/main/README.rst",
            "CPython Project Overview", "wiki", "Python Software Foundation", "PSF-2.0",
        ),
        (
            "https://raw.githubusercontent.com/microsoft/vscode/main/CONTRIBUTING.md",
            "VS Code Contributing Guide", "policy", "Microsoft", "MIT",
        ),
        (
            "https://raw.githubusercontent.com/golang/go/master/CONTRIBUTING.md",
            "Go Contributing Guide", "policy", "Go Authors", "BSD-3-Clause",
        ),
        (
            "https://raw.githubusercontent.com/golang/go/master/SECURITY.md",
            "Go Security Policy", "policy", "Go Authors", "BSD-3-Clause",
        ),
    ]

    documents = []
    for url, title, doc_type, owner, license_name in sources:
        print(f"    Downloading: {title}...")
        content = _download(url)
        if not content or len(content) < 100:
            print(f"    Skipped (too short or failed)")
            continue

        documents.append({
            "title": title,
            "text": content,
            "doc_type": doc_type,
            "owner": owner,
            "acl_roles": ["employee"],
            "authority_score": 3,
            "source_url": url,
            "license": license_name,
        })

    print(f"    Collected {len(documents)} GitHub policy docs")
    return documents


# ============================================================================
# Source 3: NIST Cybersecurity Guidance (US Government, public domain)
# ============================================================================

def collect_nist_docs() -> list[dict]:
    """
    Download NIST cybersecurity guidance documents.

    NIST publications are US Government works and therefore in the public domain.
    We download the NVD/CVE and general security guidance pages.
    """
    # NIST raw data / guidance that is available as plain text or JSON
    sources = [
        (
            "https://raw.githubusercontent.com/NIST-Cybersecurity/nist-cybersecurity-framework/main/README.md",
            "NIST Cybersecurity Framework Overview", "policy",
        ),
    ]

    documents = []
    for url, title, doc_type in sources:
        print(f"    Downloading: {title}...")
        content = _download(url)
        if not content or len(content) < 100:
            print(f"    Skipped (too short or failed)")
            continue

        documents.append({
            "title": title,
            "text": content,
            "doc_type": doc_type,
            "owner": "NIST",
            "acl_roles": ["employee", "engineering"],
            "authority_score": 5,
            "source_url": url,
            "license": "Public Domain (US Government)",
        })

    # Also try to get the NIST Password Guidelines (SP 800-63B relevant section)
    nist_password_url = "https://raw.githubusercontent.com/usnistgov/800-63-3/master/sp800-63b/sec5_authenticators.md"
    print(f"    Downloading: NIST SP 800-63B Authentication Guidelines...")
    content = _download(nist_password_url)
    if content and len(content) > 200:
        documents.append({
            "title": "NIST SP 800-63B: Authentication and Lifecycle Management",
            "text": content,
            "doc_type": "policy",
            "owner": "NIST",
            "acl_roles": ["employee", "engineering"],
            "authority_score": 5,
            "source_url": nist_password_url,
            "license": "Public Domain (US Government)",
        })

    print(f"    Collected {len(documents)} NIST docs")
    return documents


# ============================================================================
# Main: Collect, write, and generate manifest
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect public corpus documents for AHRAG")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show what would be downloaded, don't write files",
    )
    parser.add_argument(
        "--source", type=str, default="all",
        choices=["all", "kubernetes", "github", "nist"],
        help="Which source to collect from (default: all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AHRAG Public Corpus Collection")
    print("=" * 70)
    print()

    all_documents = []

    if args.source in ("all", "kubernetes"):
        print("[1/3] Kubernetes troubleshooting documentation...")
        all_documents.extend(collect_kubernetes_docs())
        print()

    if args.source in ("all", "github"):
        print("[2/3] GitHub open-source policy documents...")
        all_documents.extend(collect_github_policies())
        print()

    if args.source in ("all", "nist"):
        print("[3/3] NIST cybersecurity guidance...")
        all_documents.extend(collect_nist_docs())
        print()

    # Summary
    print("=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"  Total documents collected: {len(all_documents)}")
    total_chars = sum(len(d["text"]) for d in all_documents)
    print(f"  Total text:               {total_chars:,} chars ({total_chars/1000:.0f} KB)")
    print(f"\n  By doc type:")
    type_counts = {}
    for d in all_documents:
        type_counts[d["doc_type"]] = type_counts.get(d["doc_type"], 0) + 1
    for dt, count in sorted(type_counts.items()):
        print(f"    {dt:15s}: {count}")

    if args.dry_run:
        print(f"\nDry run — no files written.")
        for doc in all_documents:
            print(f"  [{doc['doc_type']:8s}] {doc['title']}")
            print(f"             {len(doc['text']):,} chars, license: {doc['license']}")
        return

    if not all_documents:
        print("\nNo documents collected (all downloads failed?).")
        print("Check your network connection and try again.")
        return

    # Write output
    print(f"\nWriting to {OUTPUT_DIR}/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_dir = OUTPUT_DIR / "corpus"
    corpus_dir.mkdir(exist_ok=True)

    # Write corpus files and build manifest
    manifest_docs = []
    for doc in all_documents:
        doc_id = _stable_id(doc["title"], "doc-pub")
        filename = f"{doc_id}.md"

        (corpus_dir / filename).write_text(doc["text"], encoding="utf-8")

        manifest_docs.append({
            "doc_id": doc_id,
            "path": f"corpus/{filename}",
            "title": doc["title"],
            "doc_type": doc["doc_type"],
            "owner": doc["owner"],
            "acl_roles": doc["acl_roles"],
            "created_date": "2024-01-01",
            "effective_date": "2024-01-01",
            "version": "1.0",
            "authority_score": doc["authority_score"],
        })

    # Write manifest
    manifest = {"documents": manifest_docs}
    (OUTPUT_DIR / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"  Wrote manifest.yaml ({len(manifest_docs)} documents)")
    print(f"  Wrote {len(manifest_docs)} corpus files to corpus/")

    # Write collection report
    report = {
        "total_documents": len(all_documents),
        "total_chars": total_chars,
        "sources": {
            doc["title"]: {
                "url": doc["source_url"],
                "license": doc["license"],
                "chars": len(doc["text"]),
                "doc_type": doc["doc_type"],
            }
            for doc in all_documents
        },
    }
    with (OUTPUT_DIR / "collection_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote collection_report.json")

    print(f"\nDone!")
    print(f"\nTo ingest into AHRAG:")
    print(f'  from ahrag.pipeline import AHRAGEngine')
    print(f'  engine = AHRAGEngine()')
    print(f'  engine.seed(manifest_path="{OUTPUT_DIR / "manifest.yaml"}", reset=False)')


if __name__ == "__main__":
    main()
