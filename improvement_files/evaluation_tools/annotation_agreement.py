"""Second-annotator workflow and inter-annotator agreement (improvement.txt §2b).

§2(b) asks for a second annotator to independently label 100+ queries, with
Cohen's κ or Krippendorff's α reported. That requirement cannot be satisfied by
software, and this script does not pretend otherwise:

    The labels in the benchmark evaluation set were derived programmatically
    from dataset evidence spans. A second *independent* judgement has to come
    from a second person. A model re-labelling labels it generated measures
    self-consistency, not agreement, and reporting it as κ would be worse than
    reporting nothing.

What this script does provide is everything around that judgement, so the human
effort is spent only on the judging:

  export      Draw a stratified sample and write a blank annotation sheet.
  validate    Check a returned sheet for completeness and malformed values.
  score       Compute Cohen's κ (per field) and Krippendorff's α, with
              bootstrap confidence intervals.
  adjudicate  List every disagreement, ordered so the most consequential are
              resolved first, and write an adjudicated gold set.

The sample is stratified over ``query_type`` so that the ACL-probe,
unanswerable, freshness and adversarial strata are all represented — an
agreement figure computed over 100 items that are 90% one stratum says little
about the strata that matter.

USAGE
  # 1. produce the sheet for annotator B
  python improvement_files/evaluation_tools/annotation_agreement.py export \\
      --integrated --n 120 --out annotations/sheet_B.yaml

  # 2. annotator B fills in the `label_*` fields by hand, then:
  python improvement_files/evaluation_tools/annotation_agreement.py validate \\
      --sheet annotations/sheet_B.yaml

  # 3. agreement against the existing labels
  python improvement_files/evaluation_tools/annotation_agreement.py score \\
      --integrated --sheet annotations/sheet_B.yaml

  # 4. resolve disagreements into an adjudicated set
  python improvement_files/evaluation_tools/annotation_agreement.py adjudicate \\
      --integrated --sheet annotations/sheet_B.yaml \\
      --out annotations/adjudicated.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ahrag.eval.harness import (  # noqa: E402
    add_corpus_arguments,
    build_seeded_engine,
    load_and_validate,
    resolve_corpus,
)
from ahrag.stats import bootstrap_ci  # noqa: E402

SEED = 1729

#: The fields a second annotator is asked to judge. Deliberately few: each is a
#: decision a careful reader can make from the query and the candidate evidence
#: without needing the rest of the corpus in their head.
ANNOTATION_FIELDS = {
    "label_answerable": "yes / no  — is this answerable from the shown evidence?",
    "label_should_abstain": "yes / no  — should the system refuse to answer?",
    "label_gold_sufficient": (
        "yes / no / partial  — do the shown gold chunks contain the answer?"
    ),
    "label_freshness_sensitive": (
        "yes / no  — does the correct answer depend on which version is current?"
    ),
}

VALID_VALUES = {
    "label_answerable": {"yes", "no"},
    "label_should_abstain": {"yes", "no"},
    "label_gold_sufficient": {"yes", "no", "partial"},
    "label_freshness_sensitive": {"yes", "no"},
}


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------


def cohens_kappa(a: list[str], b: list[str]) -> tuple[float, int]:
    """Cohen's κ for two annotators over the same nominal items.

    Returns ``(kappa, n)``. κ is undefined when both annotators used a single
    category — perfect agreement with no expected disagreement to correct
    for — and 1.0 is returned in that case with the caveat left to the caller.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return float("nan"), 0
    categories = sorted({v for pair in pairs for v in pair})
    index = {c: i for i, c in enumerate(categories)}
    n = len(pairs)

    matrix = np.zeros((len(categories), len(categories)), dtype=np.float64)
    for x, y in pairs:
        matrix[index[x], index[y]] += 1

    observed = float(np.trace(matrix)) / n
    row = matrix.sum(axis=1) / n
    col = matrix.sum(axis=0) / n
    expected = float((row * col).sum())
    if abs(1.0 - expected) < 1e-12:
        return 1.0, n
    return (observed - expected) / (1.0 - expected), n


def krippendorff_alpha(a: list[str], b: list[str]) -> tuple[float, int]:
    """Krippendorff's α for two annotators, nominal metric.

    Reported alongside κ because α handles missing values and generalises past
    two annotators, so it is the figure to keep reporting if a third annotator
    is ever added.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return float("nan"), 0
    n = len(pairs)

    # Observed disagreement: fraction of pairs that differ.
    observed = sum(1 for x, y in pairs if x != y) / n

    # Expected disagreement from the pooled marginal distribution.
    pooled = Counter()
    for x, y in pairs:
        pooled[x] += 1
        pooled[y] += 1
    total = sum(pooled.values())
    expected = 1.0 - sum((count / total) ** 2 for count in pooled.values())
    if expected < 1e-12:
        return 1.0, n
    return 1.0 - observed / expected, n


def interpret_kappa(value: float) -> str:
    """Landis & Koch (1977) bands. A convention, reported with the number."""
    if not np.isfinite(value):
        return "undefined"
    if value < 0.0:
        return "worse than chance"
    if value < 0.20:
        return "slight"
    if value < 0.40:
        return "fair"
    if value < 0.60:
        return "moderate"
    if value < 0.80:
        return "substantial"
    return "almost perfect"


def bootstrap_kappa(
    a: list[str], b: list[str], resamples: int = 2000
) -> tuple[float, float]:
    """Percentile bootstrap CI for κ, resampling item pairs."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if len(pairs) < 3:
        return float("nan"), float("nan")
    rng = np.random.RandomState(SEED)
    values = []
    for _ in range(resamples):
        picks = rng.randint(0, len(pairs), size=len(pairs))
        sample = [pairs[i] for i in picks]
        kappa, _ = cohens_kappa([p[0] for p in sample], [p[1] for p in sample])
        if np.isfinite(kappa):
            values.append(kappa)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ---------------------------------------------------------------------------
# Deriving annotator A's labels from the existing evaluation set
# ---------------------------------------------------------------------------


def labels_from_item(item) -> dict[str, str]:
    """Express an existing ``EvalItem`` in the annotation vocabulary.

    This is annotator A: the programmatic labelling already in the evaluation
    set, restated in the same terms the human is asked to use so the two are
    directly comparable.
    """
    return {
        "label_answerable": "no" if item.should_abstain else "yes",
        "label_should_abstain": "yes" if item.should_abstain else "no",
        "label_gold_sufficient": "yes" if item.gold_chunks else "no",
        "label_freshness_sensitive": "yes" if item.freshness_sensitive else "no",
    }


def stratified_sample(items, n: int) -> list:
    """Sample ``n`` items spread across query types, largest strata first.

    Every stratum gets at least one item where possible, then the remainder is
    allocated proportionally. Without this, a random sample of a suite that is
    34% HotpotQA would produce an agreement figure dominated by one stratum.
    """
    by_type: dict[str, list] = defaultdict(list)
    for item in items:
        by_type[item.query_type].append(item)

    rng = np.random.RandomState(SEED)
    for group in by_type.values():
        rng.shuffle(group)

    types = sorted(by_type, key=lambda t: -len(by_type[t]))
    chosen: list = []
    # One per stratum first.
    for query_type in types:
        if len(chosen) < n and by_type[query_type]:
            chosen.append(by_type[query_type].pop())
    # Then proportional to remaining stratum sizes.
    while len(chosen) < n and any(by_type[t] for t in types):
        for query_type in types:
            if len(chosen) >= n:
                break
            if by_type[query_type]:
                chosen.append(by_type[query_type].pop())
    return chosen[:n]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_export(args) -> None:
    """Write a blank annotation sheet for a second annotator."""
    manifest, eval_set = resolve_corpus(args)
    engine = build_seeded_engine(manifest, quiet=True)
    items, _ = load_and_validate(engine, eval_set, quiet=True)

    sample = stratified_sample(items, args.n)
    chunk_text = {c.chunk_id: c.text for c in engine.db.get_chunks()}

    rows = []
    for item in sample:
        evidence = [
            {
                "chunk_id": chunk_id,
                "text": chunk_text.get(chunk_id, "<not found>")[: args.excerpt],
            }
            for chunk_id in item.gold_chunks[:3]
        ]
        rows.append(
            {
                "id": item.id,
                "query": item.query,
                "asked_by": item.user_id,
                "query_type": item.query_type,
                "candidate_evidence": evidence,
                # Blank for the annotator to fill in. Annotator A's labels are
                # deliberately NOT shown: seeing them would anchor the second
                # judgement and inflate agreement.
                **{field: "" for field in ANNOTATION_FIELDS},
                "notes": "",
            }
        )

    payload = {
        "instructions": {
            "purpose": (
                "Independent second annotation for inter-annotator agreement "
                "(improvement.txt §2b). Fill in every label_* field."
            ),
            "fields": ANNOTATION_FIELDS,
            "important": (
                "Judge only from the query and the candidate evidence shown. "
                "Do not consult the existing evaluation set: this annotation is "
                "only meaningful if it is independent."
            ),
        },
        "items": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, width=100),
        encoding="utf-8",
    )

    print(f"Wrote {len(rows)} items to {out}")
    print("\nStratum coverage:")
    for query_type, count in sorted(
        Counter(r["query_type"] for r in rows).items(), key=lambda kv: -kv[1]
    ):
        print(f"  {query_type:34s} {count:4d}")
    print(f"\nNext: a second person fills in the label_* fields, then run")
    print(f"  ... annotation_agreement.py score --sheet {out}")


def command_validate(args) -> None:
    """Check a returned sheet for completeness and legal values."""
    sheet = yaml.safe_load(Path(args.sheet).read_text(encoding="utf-8"))
    rows = sheet.get("items", [])
    problems: list[str] = []
    complete = 0

    for row in rows:
        missing = [f for f in ANNOTATION_FIELDS if not str(row.get(f, "")).strip()]
        bad = [
            f"{f}={row[f]!r}"
            for f in ANNOTATION_FIELDS
            if str(row.get(f, "")).strip()
            and str(row[f]).strip().lower() not in VALID_VALUES[f]
        ]
        if bad:
            problems.append(f"{row['id']}: invalid {', '.join(bad)}")
        elif missing:
            problems.append(f"{row['id']}: missing {', '.join(missing)}")
        else:
            complete += 1

    print(f"items          : {len(rows)}")
    print(f"fully labelled : {complete}")
    print(f"problems       : {len(problems)}")
    for problem in problems[:20]:
        print(f"  {problem}")
    if len(problems) > 20:
        print(f"  ... and {len(problems) - 20} more")

    if complete < 100:
        print(
            f"\nNote: §2(b) asks for 100+ independently labelled queries; "
            f"{complete} are complete."
        )
    sys.exit(1 if problems else 0)


def command_score(args) -> None:
    """Compute agreement between the existing labels and a returned sheet."""
    manifest, eval_set = resolve_corpus(args)
    engine = build_seeded_engine(manifest, quiet=True)
    items, _ = load_and_validate(engine, eval_set, quiet=True)
    by_id = {item.id: item for item in items}

    sheet = yaml.safe_load(Path(args.sheet).read_text(encoding="utf-8"))
    rows = [r for r in sheet.get("items", []) if r["id"] in by_id]

    print("=" * 78)
    print("INTER-ANNOTATOR AGREEMENT")
    print("=" * 78)
    print("Annotator A: the programmatic labels in the evaluation set.")
    print("Annotator B: the returned sheet.")
    print()

    report: dict[str, dict] = {}
    print(f"  {'field':30s} {'n':>5s} {'agree':>7s} {'kappa':>7s} "
          f"{'95% CI':>18s}  interpretation")
    print("-" * 78)

    for field in ANNOTATION_FIELDS:
        a_labels, b_labels = [], []
        for row in rows:
            value = str(row.get(field, "")).strip().lower()
            if value not in VALID_VALUES[field]:
                continue
            a_labels.append(labels_from_item(by_id[row["id"]])[field])
            b_labels.append(value)

        if not a_labels:
            print(f"  {field:30s} {'—':>5s}  (no usable annotations)")
            continue

        raw = sum(1 for x, y in zip(a_labels, b_labels) if x == y) / len(a_labels)
        kappa, n = cohens_kappa(a_labels, b_labels)
        alpha, _ = krippendorff_alpha(a_labels, b_labels)
        low, high = bootstrap_kappa(a_labels, b_labels)
        report[field] = {
            "n": n,
            "raw_agreement": raw,
            "cohens_kappa": kappa,
            "kappa_ci95": [low, high],
            "krippendorff_alpha": alpha,
            "interpretation": interpret_kappa(kappa),
        }
        print(f"  {field:30s} {n:5d} {raw:7.3f} {kappa:7.3f} "
              f"[{low:6.3f},{high:6.3f}]  {interpret_kappa(kappa)}")

    if report:
        mean_kappa = float(
            np.mean([v["cohens_kappa"] for v in report.values()
                     if np.isfinite(v["cohens_kappa"])])
        )
        print()
        print(f"  mean kappa across fields: {mean_kappa:.3f} "
              f"({interpret_kappa(mean_kappa)})")
        alphas = [v["krippendorff_alpha"] for v in report.values()
                  if np.isfinite(v["krippendorff_alpha"])]
        if alphas:
            print(f"  mean Krippendorff alpha : {float(np.mean(alphas)):.3f}")
        total = max(v["n"] for v in report.values())
        if total < 100:
            print()
            print(f"  CAVEAT: {total} items annotated. §2(b) asks for 100+, "
                  f"and a kappa on fewer is not reportable as evidence.")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"fields": report, "items_scored": len(rows)},
                       indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nReport written to {args.output}")


def command_adjudicate(args) -> None:
    """List disagreements and write an adjudicated label set."""
    manifest, eval_set = resolve_corpus(args)
    engine = build_seeded_engine(manifest, quiet=True)
    items, _ = load_and_validate(engine, eval_set, quiet=True)
    by_id = {item.id: item for item in items}

    sheet = yaml.safe_load(Path(args.sheet).read_text(encoding="utf-8"))
    rows = [r for r in sheet.get("items", []) if r["id"] in by_id]

    disagreements = []
    for row in rows:
        item = by_id[row["id"]]
        a = labels_from_item(item)
        differing = {
            field: (a[field], str(row.get(field, "")).strip().lower())
            for field in ANNOTATION_FIELDS
            if str(row.get(field, "")).strip().lower() in VALID_VALUES[field]
            and str(row.get(field, "")).strip().lower() != a[field]
        }
        if differing:
            disagreements.append({
                "id": item.id,
                "query": item.query,
                "query_type": item.query_type,
                "asked_by": item.user_id,
                "disagreements": {
                    field: {"annotator_a": pair[0], "annotator_b": pair[1]}
                    for field, pair in differing.items()
                },
                "annotator_b_notes": row.get("notes", ""),
                "resolution": "",
                "resolution_rationale": "",
            })

    # Most consequential first: a disagreement about whether to abstain matters
    # more than one about freshness sensitivity, and items disputed on several
    # fields matter more than items disputed on one.
    weight = {
        "label_should_abstain": 4,
        "label_answerable": 3,
        "label_gold_sufficient": 2,
        "label_freshness_sensitive": 1,
    }
    disagreements.sort(
        key=lambda d: -sum(weight.get(f, 0) for f in d["disagreements"])
    )

    print(f"items compared : {len(rows)}")
    print(f"disagreements  : {len(disagreements)}")
    print()
    for entry in disagreements[:15]:
        fields = ", ".join(
            f"{f} (A={v['annotator_a']} / B={v['annotator_b']})"
            for f, v in entry["disagreements"].items()
        )
        print(f"  {entry['id'][:22]:24s} {entry['query_type']:24s} {fields}")
        print(f"    {entry['query'][:88]!r}")
    if len(disagreements) > 15:
        print(f"  ... and {len(disagreements) - 15} more")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            yaml.safe_dump(
                {
                    "instructions": (
                        "Fill in `resolution` with the agreed value and "
                        "`resolution_rationale` with why. Ordered most "
                        "consequential first."
                    ),
                    "items_compared": len(rows),
                    "disagreements": disagreements,
                },
                sort_keys=False, default_flow_style=False, width=100,
            ),
            encoding="utf-8",
        )
        print(f"\nAdjudication sheet written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Second-annotator workflow and agreement statistics",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Write a blank annotation sheet")
    add_corpus_arguments(export)
    export.add_argument("--n", type=int, default=120,
                        help="Items to sample (§2b asks for 100+)")
    export.add_argument("--excerpt", type=int, default=600,
                        help="Characters of each evidence chunk to show")
    export.add_argument("--out", type=str, default="annotations/sheet_B.yaml")
    export.set_defaults(func=command_export)

    validate = sub.add_parser("validate", help="Check a returned sheet")
    validate.add_argument("--sheet", type=str, required=True)
    validate.set_defaults(func=command_validate)

    score = sub.add_parser("score", help="Compute kappa and alpha")
    add_corpus_arguments(score)
    score.add_argument("--sheet", type=str, required=True)
    score.add_argument("--output", type=str, default=None)
    score.set_defaults(func=command_score)

    adjudicate = sub.add_parser("adjudicate", help="List and resolve disagreements")
    add_corpus_arguments(adjudicate)
    adjudicate.add_argument("--sheet", type=str, required=True)
    adjudicate.add_argument("--out", type=str, default=None)
    adjudicate.set_defaults(func=command_adjudicate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
