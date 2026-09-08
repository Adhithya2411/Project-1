"""
RAGAS Evaluation for AHRAG
============================
Runs the full AHRAG pipeline on the seeded evaluation set and evaluates
the results using RAGAS metrics (faithfulness, answer relevancy, context
precision, context recall).

This script has TWO modes:

1. --run   : Runs AHRAG pipeline, saves results JSON, then evaluates with RAGAS
2. --results FILE : Evaluates a previously-saved results JSON with RAGAS

The pipeline integration is REAL — it boots AHRAGEngine, seeds the corpus,
runs each eval query through engine.answer(), and collects the actual
generated answers and retrieved contexts.

PREREQUISITES:
  pip install ragas datasets     # For RAGAS metrics
  export OPENAI_API_KEY=your-key  # RAGAS needs an LLM judge

  If no API key is set, the script still runs the pipeline and saves the
  results JSON — you can evaluate it later when you have the key.

USAGE:
  python improvement_files/evaluation_tools/evaluate_with_ragas.py --run
  python improvement_files/evaluation_tools/evaluate_with_ragas.py --run --integrated
  python improvement_files/evaluation_tools/evaluate_with_ragas.py --results data/ragas_input.json

NOTE ON A DEFECT THIS SCRIPT USED TO HAVE:
  It called ``AHRAGEngine()`` / ``ensure_seeded()`` / ``load_eval_set()`` with no
  arguments, so it always evaluated the packaged 9-document corpus and its
  32-item suite regardless of which corpus the caller had built. It now takes
  the same ``--integrated`` / ``--manifest`` / ``--eval-set`` options as every
  other experiment script, and validates that the labels resolve against the
  corpus actually loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from ahrag.eval.harness import add_corpus_arguments, resolve_corpus  # noqa: E402


def run_ahrag_pipeline(
    output_file: str,
    manifest=None,
    eval_set=None,
    rebuild_cache: bool = False,
) -> list[dict]:
    """
    Run the full AHRAG pipeline on the seeded evaluation set and collect
    results in the format RAGAS expects.

    For each evaluation item:
      - Runs engine.answer(query, user_id) through the full 13-stage pipeline
      - Collects the generated answer text
      - Collects the retrieved evidence chunk texts as contexts
      - Collects the query and any reference answer from the eval labels

    Returns a list of dicts with keys: question, answer, contexts, ground_truth
    """
    from ahrag.eval.harness import build_seeded_engine, load_and_validate

    print("Initialising AHRAGEngine with the requested corpus...")
    engine = build_seeded_engine(manifest, rebuild_cache=rebuild_cache)

    info = engine.backend_info()
    print(f"  Engine ready: {info.get('documents', '?')} documents")
    print(f"  Generator: {info.get('generator', '?')}")
    print(f"  Router: {info.get('router', '?')}")
    print()

    items, _ = load_and_validate(engine, eval_set)
    print(f"Running pipeline on {len(items)} evaluation queries...")

    # Build a chunk-id → text lookup from all chunks in the database.
    # This is used to retrieve gold chunk texts for ground_truth references.
    all_chunks = engine.db.get_chunks()
    chunk_text_lookup = {c.chunk_id: c.text for c in all_chunks}
    print(f"  Built chunk lookup: {len(chunk_text_lookup)} chunks")
    print()

    results = []
    errors = 0

    for i, item in enumerate(items, 1):
        prefix = f"  [{i:2d}/{len(items)}] {item.id:12s}"

        try:
            started = time.perf_counter()
            response = engine.answer(item.query, item.user_id, write_audit=False)
            elapsed = time.perf_counter() - started

            # Collect the actual answer text
            answer_text = response.answer if not response.abstained else (
                f"[ABSTAINED: {response.abstention_reason.value}] "
                + (response.clarifying_question or "No answer available.")
            )

            # Collect the actual retrieved context texts
            contexts = [e.text for e in response.evidence if e.text.strip()]

            # Build ground truth from the gold chunk texts in the eval labels
            ground_truth = ""
            if item.should_abstain:
                ground_truth = "The system should abstain from answering this question."
            elif item.gold_chunks:
                gold_texts = [
                    chunk_text_lookup[cid]
                    for cid in item.gold_chunks
                    if cid in chunk_text_lookup
                ]
                ground_truth = " ".join(gold_texts) if gold_texts else ""

            results.append({
                "question": item.query,
                "answer": answer_text,
                "contexts": contexts,
                "ground_truth": ground_truth,
                # Extra metadata for analysis
                "_item_id": item.id,
                "_user_id": item.user_id,
                "_route": response.decision.route.value,
                "_abstained": response.abstained,
                "_latency_s": round(elapsed, 4),
                "_evidence_count": len(response.evidence),
                "_citation_count": len(response.citations),
            })

            route = response.decision.route.value
            status = "ABSTAINED" if response.abstained else f"{len(contexts)} ctx"
            print(f"{prefix} -> {route:3s} {status:12s} ({elapsed:.3f}s)")

        except Exception as exc:
            errors += 1
            print(f"{prefix} -> ERROR: {exc}")
            results.append({
                "question": item.query,
                "answer": f"[ERROR: {exc}]",
                "contexts": [],
                "ground_truth": "",
                "_item_id": item.id,
                "_user_id": item.user_id,
                "_route": "ERROR",
                "_abstained": True,
                "_latency_s": 0.0,
                "_evidence_count": 0,
                "_citation_count": 0,
            })

    print()
    print(f"Pipeline complete: {len(results)} results, {errors} errors")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to: {output_path}")

    # Print summary stats
    routes = {}
    for r in results:
        route = r.get("_route", "?")
        routes[route] = routes.get(route, 0) + 1
    print(f"\nRoute distribution: {routes}")
    abstained = sum(1 for r in results if r.get("_abstained"))
    print(f"Abstained: {abstained}/{len(results)}")
    avg_ctx = sum(len(r["contexts"]) for r in results) / max(1, len(results))
    print(f"Avg contexts per query: {avg_ctx:.1f}")

    return results


def evaluate_with_ragas(results: list[dict]) -> dict | None:
    """
    Evaluate AHRAG results using RAGAS metrics.

    Requires: pip install ragas datasets
    Requires: OPENAI_API_KEY environment variable set

    Returns the RAGAS scores dict, or None if RAGAS is unavailable.
    """
    # Check for API key first
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print()
        print("=" * 60)
        print("RAGAS EVALUATION SKIPPED — No OpenAI API key")
        print("=" * 60)
        print()
        print("RAGAS requires an LLM judge (OpenAI by default) to compute")
        print("faithfulness and answer relevancy scores.")
        print()
        print("To run RAGAS evaluation:")
        print("  1. Set your API key: set OPENAI_API_KEY=your-key")
        print("  2. Re-run with: python evaluate_with_ragas.py --results <file>")
        print()
        print("The pipeline results have been saved — you can evaluate them")
        print("at any time without re-running the AHRAG pipeline.")
        return None

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        print()
        print("=" * 60)
        print("RAGAS EVALUATION SKIPPED — Missing packages")
        print("=" * 60)
        print()
        print("Install required packages:")
        print("  pip install ragas datasets")
        return None

    # Filter out entries with no contexts (abstentions/errors)
    evaluable = [r for r in results if r["contexts"] and r["answer"]]
    if not evaluable:
        print("No evaluable results (all abstentions or errors)")
        return None

    print()
    print(f"Running RAGAS evaluation on {len(evaluable)} queries...")
    print(f"(Skipping {len(results) - len(evaluable)} abstentions/errors)")
    print()

    # Format for RAGAS — strip underscore-prefixed metadata keys
    data = {
        "question": [r["question"] for r in evaluable],
        "answer": [r["answer"] for r in evaluable],
        "contexts": [r["contexts"] for r in evaluable],
        "ground_truth": [r.get("ground_truth", "") for r in evaluable],
    }

    dataset = Dataset.from_dict(data)

    # Select metrics based on available data
    metrics = [faithfulness, answer_relevancy, context_precision]
    has_ground_truth = any(r.get("ground_truth", "").strip() for r in evaluable)
    if has_ground_truth:
        metrics.append(context_recall)

    result = evaluate(dataset=dataset, metrics=metrics)

    print()
    print("=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    for metric_name, score in result.items():
        if isinstance(score, (int, float)):
            print(f"  {metric_name:25s}: {score:.4f}")
    print("=" * 60)

    return dict(result)


def main():
    parser = argparse.ArgumentParser(
        description="Run AHRAG pipeline and evaluate with RAGAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the full pipeline and save results (works without API key):
  python evaluate_with_ragas.py --run

  # Run pipeline and save to a specific file:
  python evaluate_with_ragas.py --run --output my_results.json

  # Evaluate previously-saved results with RAGAS:
  python evaluate_with_ragas.py --results my_results.json
        """,
    )
    add_corpus_arguments(parser)
    parser.add_argument(
        "--run", action="store_true",
        help="Run AHRAG pipeline on seeded eval set, save results, then evaluate",
    )
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to previously-saved results JSON to evaluate with RAGAS",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "data" / "ragas_input.json"),
        help="Where to save pipeline results (default: data/ragas_input.json)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AHRAG × RAGAS Evaluation")
    print("=" * 60)
    print()

    if not args.run and not args.results:
        # Default: run the pipeline
        print("No mode specified. Running --run by default.")
        print()
        args.run = True

    if args.run:
        # Run the full AHRAG pipeline
        manifest, eval_set = resolve_corpus(args)
        results = run_ahrag_pipeline(
            args.output, manifest, eval_set, args.rebuild_cache
        )
        print()

        # Try RAGAS evaluation (will skip gracefully if no API key)
        ragas_scores = evaluate_with_ragas(results)

        if ragas_scores:
            # Save RAGAS scores alongside results
            scores_file = args.output.replace(".json", "_ragas_scores.json")
            with open(scores_file, "w") as f:
                json.dump(ragas_scores, f, indent=2, default=str)
            print(f"\nRAGAS scores saved to: {scores_file}")

    elif args.results:
        # Load and evaluate existing results
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"ERROR: Results file not found: {results_path}")
            sys.exit(1)

        with results_path.open() as f:
            results = json.load(f)

        print(f"Loaded {len(results)} results from {results_path}")
        ragas_scores = evaluate_with_ragas(results)

        if ragas_scores:
            scores_file = str(results_path).replace(".json", "_ragas_scores.json")
            with open(scores_file, "w") as f:
                json.dump(ragas_scores, f, indent=2, default=str)
            print(f"\nRAGAS scores saved to: {scores_file}")


if __name__ == "__main__":
    main()
