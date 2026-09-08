"""Prepare a reproducible, evidence-linked FinanceBench AHRAG subset.

Usage:
    python improvement_files/datasets/prepare_financebench.py --documents 30

The script selects the documents referenced by the most open-source questions,
extracts their text, and writes an AHRAG manifest plus evaluation YAML. It does
not copy PDFs or alter the application's seed corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from ahrag.ingestion.chunker import chunk_document
from ahrag.models import DocType, DocumentMeta


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "improvement_files" / "datasets"
    pdf_dir = data_dir / "financebench_repo" / "pdfs"
    rows = _load_rows(data_dir / "financebench_open_source.jsonl")
    pdfs = {path.stem: path for path in pdf_dir.glob("*.pdf")}
    counts = Counter(row["doc_name"] for row in rows if row["doc_name"] in pdfs)
    selected = [name for name, _ in counts.most_common(args.documents)]
    if not selected:
        raise SystemExit("No FinanceBench questions matched available PDFs.")

    output_dir = data_dir / "financebench_subset"
    output_dir.mkdir(exist_ok=True)
    manifest_entries = []
    chunks_by_doc: dict[str, list] = {}
    for ordinal, doc_name in enumerate(selected, start=1):
        source = pdfs[doc_name]
        evidence_pages = []
        seen_pages = set()
        for row in rows:
            if row["doc_name"] != doc_name:
                continue
            for evidence in row.get("evidence", []):
                page = evidence.get("evidence_text_full_page", "").strip()
                page_key = (evidence.get("evidence_page_num"), page)
                if page and page_key not in seen_pages:
                    seen_pages.add(page_key)
                    evidence_pages.append(page)
        text = "\n\n".join(evidence_pages)
        if not text:
            continue
        doc_id = f"doc-financebench-{hashlib.sha256(doc_name.encode()).hexdigest()[:12]}"
        meta = DocumentMeta(
            doc_id=doc_id, title=doc_name.replace("_", " "), source_uri=str(source),
            doc_type=DocType.FINANCE, owner="financebench",
            acl_roles=["finance", "admin"], created_date=date(2022, 1, 1),
            effective_date=date(2022, 1, 1), version="1.0", authority_score=5,
            checksum=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        )
        (output_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        chunks_by_doc[doc_name] = chunk_document(text, meta)
        manifest_entries.append({
            "doc_id": doc_id, "path": f"{doc_id}.txt",
            "title": doc_name.replace("_", " "), "doc_type": "finance",
            "owner": "financebench", "acl_roles": ["finance", "admin"],
            "created_date": "2022-01-01", "effective_date": "2022-01-01",
            "version": "1.0", "authority_score": 5,
        })

    selected_rows = [row for row in rows if row["doc_name"] in selected]
    eval_items = []
    for index, row in enumerate(selected_rows, start=1):
        gold_chunks = []
        evidence = row.get("evidence", [])
        for evidence_item in evidence:
            target = _normalise(evidence_item.get("evidence_text", ""))
            if not target:
                continue
            for chunk in chunks_by_doc.get(evidence_item.get("evidence_doc_name", row["doc_name"]), []):
                if target in _normalise(chunk.text):
                    gold_chunks.append(chunk.chunk_id)
        gold_chunks = list(dict.fromkeys(gold_chunks))
        if not gold_chunks:
            continue
        eval_items.append({
            "id": f"financebench-{row['financebench_id']}",
            "query": row["question"], "user_id": "finance.analyst",
            "query_type": f"finance_{row.get('question_type', 'unknown')}",
            "expected_route": "R3", "gold_chunks": gold_chunks,
            "should_abstain": False, "freshness_sensitive": False,
            "notes": "Evidence-linked from FinanceBench open-source annotation.",
        })

    manifest = {
        "users": [{
            "user_id": "finance.analyst", "display_name": "Finance analyst",
            "roles": ["finance"],
        }],
        "documents": manifest_entries,
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "eval_set.yaml").write_text(
        yaml.safe_dump({"items": eval_items}, sort_keys=False), encoding="utf-8"
    )
    print(f"Prepared {len(manifest_entries)} documents and {len(eval_items)} evidence-linked questions.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()