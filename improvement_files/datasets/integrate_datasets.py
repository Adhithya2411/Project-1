"""
Dataset Integration for AHRAG
================================
Converts downloaded benchmark datasets into AHRAG-compatible evaluation
items and corpus documents.

This is NOT a template — it parses the actual downloaded JSON/JSONL files
in improvement_files/datasets/ and produces valid AHRAG manifest.yaml and
eval_set.yaml files that can be loaded by ahrag.eval.dataset.load_eval_set()
and ahrag.ingestion.pipeline.IngestionService.seed_from_manifest().

WHAT THIS SCRIPT DOES:

1. FinanceBench (financebench_hf.json):
   - Extracts evidence text from each question's evidence spans
   - Groups by document (SEC 10-K filings) to create corpus documents
   - Creates evaluation items with evidence-linked gold chunks
   - Generates proper ACL roles, authority scores, doc types

2. PolicyQA (policyqa_train.json):
   - Extracts unique policy document contexts
   - Creates evaluation items from questions + context spans
   - Assigns 'policy' doc_type with appropriate ACL roles

3. HR Policies QA (hr_policies_qa.json):
   - Extracts Q&A pairs from chat-format data
   - Creates evaluation queries (no source documents — eval only)

4. HotpotQA (hotpotqa_dev.json):
   - Extracts supporting facts as document content
   - Creates multi-hop evaluation items with R4-appropriate routing

OUTPUT:
   improvement_files/datasets/integrated/
     ├── manifest.yaml          (corpus manifest for AHRAG ingestion)
     ├── eval_set.yaml          (evaluation items for AHRAG evaluation)
     ├── corpus/                (extracted document text files)
     └── integration_report.json (statistics and metadata)

USAGE:
  python improvement_files/datasets/integrate_datasets.py
  python improvement_files/datasets/integrate_datasets.py --preview
  python improvement_files/datasets/integrate_datasets.py --dataset financebench
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS_DIR = PROJECT_ROOT / "improvement_files" / "datasets"
OUTPUT_DIR = DATASETS_DIR / "integrated"


def _stable_id(text: str, prefix: str) -> str:
    """Generate a stable, deterministic ID from content."""
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def _chunk_id(doc_id: str, ordinal: int) -> str:
    """Generate a chunk ID matching AHRAG's convention: doc_id::c{ordinal:03d}"""
    return f"{doc_id}::c{ordinal:03d}"


# ============================================================================
# FinanceBench Integration
# ============================================================================

def integrate_financebench() -> tuple[list[dict], list[dict], list[dict]]:
    """
    Parse financebench_hf.json and produce AHRAG-compatible documents and
    evaluation items.

    Returns: (documents, eval_items, corpus_files)
        documents: list of manifest document entries
        eval_items: list of eval_set items
        corpus_files: list of (filename, text) tuples to write
    """
    hf_path = DATASETS_DIR / "financebench_hf.json"
    if not hf_path.exists():
        print("  financebench_hf.json not found, skipping")
        return [], [], []

    print(f"  Loading {hf_path.name}...")
    rows = []
    with hf_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not rows:
        print("  No rows loaded")
        return [], [], []

    print(f"  Loaded {len(rows)} FinanceBench questions")

    # Group evidence by document name to create corpus documents
    doc_evidence: dict[str, list[str]] = defaultdict(list)
    doc_metadata: dict[str, dict] = {}

    for row in rows:
        doc_name = row.get("doc_name", "")
        if not doc_name:
            continue

        # Collect evidence pages
        for ev in row.get("evidence", []):
            page_text = ev.get("evidence_text_full_page", "").strip()
            if page_text and page_text not in doc_evidence[doc_name]:
                doc_evidence[doc_name].append(page_text)

        # Store doc metadata
        if doc_name not in doc_metadata:
            doc_metadata[doc_name] = {
                "company": row.get("company", "Unknown"),
                "doc_type": row.get("doc_type", "10k"),
                "doc_period": row.get("doc_period", 2022),
                "sector": row.get("gics_sector", ""),
            }

    # Create AHRAG documents from the grouped evidence
    documents = []
    corpus_files = []

    for doc_name, pages in doc_evidence.items():
        if not pages:
            continue

        meta = doc_metadata.get(doc_name, {})
        doc_id = _stable_id(doc_name, "doc-fb")
        title = doc_name.replace("_", " ")
        text = "\n\n---\n\n".join(pages)

        corpus_files.append((f"{doc_id}.txt", text))

        period = meta.get("doc_period", 2022)
        documents.append({
            "doc_id": doc_id,
            "path": f"corpus/{doc_id}.txt",
            "title": title,
            "doc_type": "finance",
            "owner": meta.get("company", "Unknown"),
            "acl_roles": ["finance", "employee"],
            "created_date": f"{period}-01-01",
            "effective_date": f"{period}-01-01",
            "version": "1.0",
            "authority_score": 4,
        })

    # Create evaluation items from questions
    eval_items = []
    for row in rows:
        doc_name = row.get("doc_name", "")
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        fb_id = row.get("financebench_id", "")

        if not question or not doc_name:
            continue

        doc_id = _stable_id(doc_name, "doc-fb")

        # Find which chunk the evidence text maps to
        # We use the evidence text to match against the document pages
        gold_chunks = []
        evidence_texts = []
        for ev in row.get("evidence", []):
            ev_text = ev.get("evidence_text", "").strip().lower()
            if not ev_text:
                continue
            evidence_texts.append(ev_text)

            # Match evidence against the document's pages to find chunk ordinal
            pages = doc_evidence.get(doc_name, [])
            for ordinal, page in enumerate(pages):
                if ev_text[:100] in page.lower():
                    chunk_id = _chunk_id(doc_id, ordinal)
                    if chunk_id not in gold_chunks:
                        gold_chunks.append(chunk_id)
                    break

        if not gold_chunks:
            # Fallback: use first chunk of the document
            gold_chunks = [_chunk_id(doc_id, 0)]

        q_type = str(row.get("question_type") or "unknown")
        reasoning = str(row.get("question_reasoning") or "")

        # Map FinanceBench question types to expected routes
        if "numerical" in reasoning.lower() or "calculation" in reasoning.lower():
            expected_route = "R3"  # Needs hybrid for numerical + context
        elif "logical" in reasoning.lower():
            expected_route = "R4"  # Multi-step reasoning
        elif "information extraction" in reasoning.lower():
            expected_route = "R1"  # Direct lookup
        else:
            expected_route = "R3"  # Default to hybrid

        eval_items.append({
            "id": f"fb-{fb_id}" if fb_id else _stable_id(question, "fb"),
            "query": question,
            "user_id": "finance.analyst",
            "query_type": f"finance_{q_type}",
            "expected_route": expected_route,
            "gold_chunks": gold_chunks,
            "should_abstain": False,
            "freshness_sensitive": False,
            "notes": f"FinanceBench {q_type}. Reference answer: {answer[:100]}...",
        })

    print(f"  Created {len(documents)} documents, {len(eval_items)} eval items")
    return documents, eval_items, corpus_files


# ============================================================================
# PolicyQA Integration
# ============================================================================

def integrate_policyqa() -> tuple[list[dict], list[dict], list[dict]]:
    """
    Parse policyqa_train.json and produce AHRAG-compatible documents and
    evaluation items.

    PolicyQA contains reading comprehension items over privacy policy documents.
    Each item has a context (policy paragraph), question, and answer span.
    """
    pqa_path = DATASETS_DIR / "policyqa_train.json"
    if not pqa_path.exists():
        print("  policyqa_train.json not found, skipping")
        return [], [], []

    print(f"  Loading {pqa_path.name}...")
    rows = []
    with pqa_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not rows:
        print("  No rows loaded")
        return [], [], []

    print(f"  Loaded {len(rows)} PolicyQA items")

    # Group by unique context text to create documents
    # PolicyQA has many questions per policy paragraph
    context_groups: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        context = row.get("context", "").strip()
        if not context or len(context) < 50:
            continue
        # Use first 200 chars as key to group similar contexts
        context_key = context[:200]
        context_groups[context_key].append(row)

    # Limit to top 50 most question-rich contexts (for manageable size)
    sorted_groups = sorted(context_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    selected = sorted_groups[:50]

    documents = []
    eval_items = []
    corpus_files = []

    for context_key, group_rows in selected:
        context = group_rows[0].get("context", "")
        doc_id = _stable_id(context_key, "doc-pqa")
        title = f"Privacy Policy Section ({doc_id[-8:]})"

        corpus_files.append((f"{doc_id}.txt", context))

        documents.append({
            "doc_id": doc_id,
            "path": f"corpus/{doc_id}.txt",
            "title": title,
            "doc_type": "policy",
            "owner": "Privacy Team",
            "acl_roles": ["employee", "hr", "manager"],
            "created_date": "2023-01-01",
            "effective_date": "2023-01-01",
            "version": "1.0",
            "authority_score": 4,
        })

        # Create eval items (max 3 per document to keep size manageable)
        for qi, row in enumerate(group_rows[:3]):
            question = row.get("question", "").strip()
            if not question:
                continue

            # Answer span tells us the gold chunk
            answers = row.get("answers", {})
            answer_text = ""
            if isinstance(answers, dict):
                answer_texts = answers.get("text", [])
                if answer_texts:
                    answer_text = answer_texts[0]

            eval_items.append({
                "id": _stable_id(f"{doc_id}-{qi}-{question}", "pqa"),
                "query": question,
                "user_id": "alice.employee",
                "query_type": "policy_comprehension",
                "expected_route": "R2",  # Semantic comprehension
                "gold_chunks": [_chunk_id(doc_id, 0)],
                "should_abstain": False,
                "freshness_sensitive": False,
                "notes": f"PolicyQA. Answer: {answer_text[:80]}",
            })

    print(f"  Created {len(documents)} documents, {len(eval_items)} eval items")
    return documents, eval_items, corpus_files


# ============================================================================
# HR Policies QA Integration (evaluation-only, no source documents)
# ============================================================================

def integrate_hr_policies_qa() -> list[dict]:
    """
    Parse hr_policies_qa.json and produce evaluation-only items.

    HR Policies QA is in chat-completion format with no source documents,
    so we can only create evaluation queries — not corpus documents.
    These items should be marked should_abstain=True when run against the
    seeded AHRAG corpus (since the content isn't ingested), or used as
    additional training data for query feature extraction.
    """
    hr_path = DATASETS_DIR / "hr_policies_qa.json"
    if not hr_path.exists():
        print("  hr_policies_qa.json not found, skipping")
        return []

    print(f"  Loading {hr_path.name}...")
    rows = []
    with hr_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not rows:
        print("  No rows loaded")
        return []

    print(f"  Loaded {len(rows)} HR Policies QA items")

    eval_items = []
    seen_questions = set()

    for row in rows:
        messages = row.get("messages", [])
        question = ""
        answer = ""
        for msg in messages:
            if msg.get("role") == "user":
                question = msg.get("content", "").strip()
            elif msg.get("role") == "assistant":
                answer = msg.get("content", "").strip()

        if not question or question in seen_questions:
            continue
        seen_questions.add(question)

        # These are HR questions — they MIGHT be answerable from the seeded
        # AHRAG corpus (which has HR leave policies). Check if topic matches.
        hr_topics = ["leave", "holiday", "annual leave", "absence", "sick",
                      "working", "remote", "hybrid", "expense", "travel"]
        topic_match = any(t in question.lower() for t in hr_topics)

        eval_items.append({
            "id": _stable_id(question, "hrqa"),
            "query": question,
            "user_id": "carol.hr",
            "query_type": "hr_policy",
            "expected_route": "R2",
            "gold_chunks": [],  # No gold chunks — eval-only
            "should_abstain": not topic_match,  # Abstain if topic doesn't match corpus
            "freshness_sensitive": False,
            "notes": f"HR Policies QA. Topic match: {topic_match}. Answer: {answer[:80]}",
        })

    # Limit to 100 items for manageability
    eval_items = eval_items[:100]
    print(f"  Created {len(eval_items)} eval-only items (no corpus documents)")
    return eval_items


# ============================================================================
# HotpotQA Integration
# ============================================================================

def integrate_hotpotqa(max_items: int = 100) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Parse hotpotqa_dev.json and produce AHRAG-compatible documents and
    multi-hop evaluation items.

    HotpotQA provides supporting facts from Wikipedia paragraphs for each
    question. We extract these as mini-documents and create evaluation items
    that test AHRAG's R4 decomposed iterative retrieval.
    """
    hotpot_path = DATASETS_DIR / "hotpotqa_dev.json"
    if not hotpot_path.exists():
        print("  hotpotqa_dev.json not found, skipping")
        return [], [], []

    print(f"  Loading {hotpot_path.name} (first {max_items} items)...")
    rows = []
    with hotpot_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_items:
                break
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not rows:
        print("  No rows loaded")
        return [], [], []

    print(f"  Loaded {len(rows)} HotpotQA items")

    # Collect unique Wikipedia article texts
    article_texts: dict[str, str] = {}

    for row in rows:
        context = row.get("context", {})
        if isinstance(context, dict):
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            for title, sentences in zip(titles, sentences_list):
                if title not in article_texts:
                    article_texts[title] = " ".join(sentences)
        elif isinstance(context, list):
            # Alternative format: list of [title, sentences] pairs
            for item in context:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    title, sentences = item[0], item[1]
                    if title not in article_texts:
                        article_texts[title] = " ".join(sentences) if isinstance(sentences, list) else str(sentences)

    # Create AHRAG documents from Wikipedia articles
    documents = []
    corpus_files = []
    title_to_doc_id: dict[str, str] = {}

    for title, text in article_texts.items():
        if not text or len(text) < 50:
            continue

        doc_id = _stable_id(title, "doc-hqa")
        title_to_doc_id[title] = doc_id

        corpus_files.append((f"{doc_id}.txt", text))

        documents.append({
            "doc_id": doc_id,
            "path": f"corpus/{doc_id}.txt",
            "title": f"Wikipedia: {title}",
            "doc_type": "wiki",
            "owner": "Wikipedia",
            "acl_roles": ["employee"],
            "created_date": "2023-01-01",
            "effective_date": "2023-01-01",
            "version": "1.0",
            "authority_score": 3,
        })

    # Create evaluation items
    eval_items = []

    for row in rows:
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        q_type = row.get("type", "")
        level = row.get("level", "")
        supporting_facts = row.get("supporting_facts", {})

        if not question:
            continue

        # Find gold chunks from supporting facts
        gold_chunks = []
        if isinstance(supporting_facts, dict):
            sf_titles = supporting_facts.get("title", [])
        elif isinstance(supporting_facts, list):
            sf_titles = [sf[0] for sf in supporting_facts if isinstance(sf, (list, tuple))]
        else:
            sf_titles = []

        for sf_title in sf_titles:
            doc_id = title_to_doc_id.get(sf_title, "")
            if doc_id:
                chunk_id = _chunk_id(doc_id, 0)
                if chunk_id not in gold_chunks:
                    gold_chunks.append(chunk_id)

        if not gold_chunks:
            continue

        # Determine expected route
        if q_type == "comparison" or len(gold_chunks) >= 2:
            expected_route = "R4"  # Multi-hop / comparison
        elif level == "hard":
            expected_route = "R3"  # Hard → hybrid
        else:
            expected_route = "R3"  # Default hybrid

        eval_items.append({
            "id": _stable_id(question, "hqa"),
            "query": question,
            "user_id": "alice.employee",
            "query_type": f"hotpotqa_{q_type}" if q_type else "hotpotqa",
            "expected_route": expected_route,
            "gold_chunks": gold_chunks,
            "should_abstain": False,
            "freshness_sensitive": False,
            "notes": f"HotpotQA {q_type}/{level}. Answer: {answer[:60]}",
        })

    print(f"  Created {len(documents)} documents, {len(eval_items)} eval items")
    return documents, eval_items, corpus_files


# ============================================================================
# Main: Combine all datasets and write output
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Integrate downloaded datasets into AHRAG format",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Only show what would be generated, don't write files",
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all", "financebench", "policyqa", "hr", "hotpotqa"],
        help="Which dataset to integrate (default: all)",
    )
    parser.add_argument(
        "--hotpotqa-limit", type=int, default=100,
        help="Max HotpotQA items to include (default: 100)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AHRAG Dataset Integration")
    print("=" * 70)
    print()

    all_documents = []
    all_eval_items = []
    all_corpus_files = []
    all_eval_only_items = []

    # FinanceBench
    if args.dataset in ("all", "financebench"):
        print("[1/4] FinanceBench...")
        docs, evals, files = integrate_financebench()
        all_documents.extend(docs)
        all_eval_items.extend(evals)
        all_corpus_files.extend(files)
        print()

    # PolicyQA
    if args.dataset in ("all", "policyqa"):
        print("[2/4] PolicyQA...")
        docs, evals, files = integrate_policyqa()
        all_documents.extend(docs)
        all_eval_items.extend(evals)
        all_corpus_files.extend(files)
        print()

    # HR Policies QA (eval-only)
    if args.dataset in ("all", "hr"):
        print("[3/4] HR Policies QA...")
        hr_evals = integrate_hr_policies_qa()
        all_eval_only_items.extend(hr_evals)
        print()

    # HotpotQA
    if args.dataset in ("all", "hotpotqa"):
        print("[4/4] HotpotQA...")
        docs, evals, files = integrate_hotpotqa(max_items=args.hotpotqa_limit)
        all_documents.extend(docs)
        all_eval_items.extend(evals)
        all_corpus_files.extend(files)
        print()

    # Add a finance.analyst user for FinanceBench queries
    users = [
        {
            "user_id": "finance.analyst",
            "display_name": "Finance Analyst (benchmark)",
            "roles": ["finance", "employee"],
            "description": "Benchmark user for FinanceBench evaluation.",
        },
    ]

    # Summary
    print("=" * 70)
    print("INTEGRATION SUMMARY")
    print("=" * 70)
    print(f"  Corpus documents:      {len(all_documents)}")
    print(f"  Corpus text files:     {len(all_corpus_files)}")
    print(f"  Eval items (with gold): {len(all_eval_items)}")
    print(f"  Eval items (eval-only): {len(all_eval_only_items)}")
    total_chars = sum(len(text) for _, text in all_corpus_files)
    print(f"  Total corpus text:     {total_chars:,} chars ({total_chars/1000:.0f} KB)")

    # Count by type
    type_counts = defaultdict(int)
    for item in all_eval_items:
        type_counts[item["query_type"]] += 1
    print(f"\n  Eval items by type:")
    for qt, count in sorted(type_counts.items()):
        print(f"    {qt:30s}: {count}")

    if args.preview:
        print(f"\nPreview mode — no files written.")
        if all_documents:
            print(f"\nSample document entry:")
            print(yaml.safe_dump(all_documents[0], sort_keys=False, default_flow_style=False))
        if all_eval_items:
            print(f"Sample eval item:")
            print(yaml.safe_dump(all_eval_items[0], sort_keys=False, default_flow_style=False))
        return

    # Write output files
    print(f"\nWriting output to {OUTPUT_DIR}/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_dir = OUTPUT_DIR / "corpus"
    corpus_dir.mkdir(exist_ok=True)

    # Write corpus text files
    for filename, text in all_corpus_files:
        (corpus_dir / filename).write_text(text, encoding="utf-8")
    print(f"  Wrote {len(all_corpus_files)} corpus files")

    # Write manifest.yaml
    manifest = {
        "users": users,
        "documents": all_documents,
    }
    (OUTPUT_DIR / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"  Wrote manifest.yaml ({len(all_documents)} documents)")

    # Write eval_set.yaml (items with gold chunks)
    eval_set = {"items": all_eval_items}
    (OUTPUT_DIR / "eval_set.yaml").write_text(
        yaml.safe_dump(eval_set, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"  Wrote eval_set.yaml ({len(all_eval_items)} items)")

    # Write eval-only items (no gold chunks, for extended testing)
    if all_eval_only_items:
        eval_only = {"items": all_eval_only_items}
        (OUTPUT_DIR / "eval_set_extended.yaml").write_text(
            yaml.safe_dump(eval_only, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(f"  Wrote eval_set_extended.yaml ({len(all_eval_only_items)} eval-only items)")

    # Write integration report
    report = {
        "datasets_integrated": args.dataset,
        "total_documents": len(all_documents),
        "total_eval_items": len(all_eval_items),
        "total_eval_only_items": len(all_eval_only_items),
        "total_corpus_chars": total_chars,
        "eval_types": dict(type_counts),
    }
    with (OUTPUT_DIR / "integration_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote integration_report.json")

    print(f"\nDone! Output directory: {OUTPUT_DIR}")
    print(f"\nTo ingest into AHRAG:")
    print(f"  from ahrag.pipeline import AHRAGEngine")
    print(f'  engine = AHRAGEngine()')
    print(f'  engine.seed(manifest_path="{OUTPUT_DIR / "manifest.yaml"}")')


if __name__ == "__main__":
    main()
