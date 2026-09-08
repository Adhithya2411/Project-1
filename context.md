# AHRAG — Project Context and Status

Context-injection document for a future session or a developer joining the
project. It is written to be pasted or read first, and to be *accurate* rather
than encouraging: the previous version of this file described work as complete
that was not, and a session that trusts it will draw wrong conclusions.

Companion documents, in the order worth reading them:

| File | What it is for |
|---|---|
| `FINDINGS.md` | Every measured result, including the negative ones. Start here for numbers. |
| `RESEARCH_LIMITATIONS.md` | What the project does **not** establish, and what would be needed. |
| `INVENTION_DISCLOSURE.md` | Mechanisms that might be differentiating, with prior-art risk, plus a correction to M1. |
| `README.md` | How to run it; architecture walkthrough. |
| `improvement.txt` | The original review that set the current work programme. |

---

## 1. What AHRAG is

A governance-aware adaptive router for enterprise RAG. The organising claim:

> Route selection is constrained by *authorised source scope* and optimised for
> evidence sufficiency, freshness, source authority, estimated latency and
> estimated cost — not merely query complexity.

Three behaviours follow from that, and they are what the tests and evaluation
actually check:

1. **Authorisation is a hard pre-filter, not a scoring term.** A per-principal
   `AuthorisedScope` is computed *before* routing, and every retriever is
   restricted to it. Cost-optimal routing cannot trade governance for latency,
   because that trade is not representable.
2. **Adaptive routing.** Five routes — R0 abstain, R1 sparse, R2 dense,
   R3 hybrid RRF, R4 decomposed iterative — selected per query.
3. **Abstention and disclosure over hallucination.** Insufficient or
   contradictory evidence produces a *typed* refusal or a clarifying question,
   and version conflicts are disclosed rather than silently resolved.

---

## 2. Architecture

### The routing decision is two-layer

This is the part worth understanding first, because it is where the design
claim lives:

```
  ACL pre-filter  ->  AuthorisedScope        (hard, upstream, not a utility term)
        |
        v
  admissible route set   <-  governance predicates
        |                    (empty scope; probe confidence below floor)
        v
  ranking, over admissible routes only:
        P1  hand-tuned utility   U(z|x) = Q - λ_L·L - λ_C·C - λ_R·R
        P2  trained classifier   (23 features, incl. governance + probe)
        B6  trained classifier   (18 text-only features = Adaptive-RAG)
```

Admissibility is computed before any preference is evaluated, so no value of
the quality, latency or cost terms can reinstate an excluded route. The learned
router changes *which admissible route wins*; it cannot change *what is
permitted*.

### Core modules

| File | Role |
|---|---|
| `ahrag/pipeline.py` | `AHRAGEngine` — the 11-stage lifecycle. Entry point is `engine.answer(query, user_id)`. |
| `ahrag/governance/acl.py` | The hard ACL pre-filter; produces `AuthorisedScope`. |
| `ahrag/routing/router.py` | `GovernanceAwareRouter` (P1), `LearnedRouter` (P2), `AdaptiveRAGRouter` (B6), `ComplexityOnlyRouter` (B5), `FixedRouter` (B1–B4), `build_router`. |
| `ahrag/routing/features.py` | The 23 router features. |
| `ahrag/retrieval/pipeline.py` | Per-route retrieval, all ACL-scoped. |
| `ahrag/evidence/` | Sufficiency gate and conflict detection. |
| `ahrag/index/` | BM25, embeddings, vector store, **and the ACL lattice / scope specialisation**. |
| `ahrag/eval/systems.py` | The eight comparable systems. |
| `ahrag/eval/harness.py` | Corpus + eval-set selection, label validation, corpus cache. |
| `ahrag/stats.py` | Bootstrap CIs, paired tests, Cohen's *d*. |
| `ahrag/models.py` | All Pydantic domain schemas. |
| `ahrag/db.py` | SQLite storage for documents, chunks, ACL metadata, audit log. |

### Modules added in the most recent session

| File | Role |
|---|---|
| `ahrag/index/lattice.py` | ACL equivalence classes — the corpus as a lattice of subcorpora. |
| `ahrag/index/scoped.py` | Scope-Pure Index Specialisation: per-class BM25 IDF, LSA basis, reranker. |
| `ahrag/stats.py` | Statistics, shared by every experiment script. |
| `ahrag/eval/harness.py` | Shared corpus/eval-set plumbing. |
| `ahrag/_openmp.py` | Ordered optional-dependency load; prevents a hard process abort. |

---

## 3. Two corpora, and which one a result came from

**Always state which corpus a number came from.** Conflating them is how the
previous phase produced misleading results.

| | Demo corpus | Benchmark corpus |
|---|---|---|
| Documents | 9 | **6,139** (249 enterprise-shaped + 5,890 reference paragraphs) |
| Chunks | 60 | **7,082** |
| Eval items | 32 | **965** |
| ACL classes | 4 | **7** |
| Supersession chains | 1 | **15** |
| Built by | packaged in `ahrag/seed/` | `improvement_files/datasets/integrate_datasets.py` |
| Selected with | default | `--integrated` |

Every experiment script accepts `--integrated`, or `--manifest` / `--eval-set`
for an arbitrary corpus, and **validates that the evaluation labels resolve
against the corpus actually loaded**. That check exists because its absence
previously allowed 83% of gold chunk IDs to point at the wrong text while every
metric read out as a plausible-looking low score.

The benchmark corpus is ingested once and cached under `data/corpus-cache/`,
keyed by manifest content. First build is ~4 minutes; subsequent runs are ~14
seconds.

---

## 4. Current status

**Working and verified.** 290 tests pass. All 9 API endpoints return 200. The
Streamlit UI runs clean under `AppTest`. ACL enforcement, conflict disclosure
and audit content-minimisation are verified end to end over HTTP, not only in
unit tests.

**Established results.** See `FINDINGS.md` for the full set. The load-bearing
ones:

- The learned router reaches **0.783** held-out route accuracy against
  **0.566** for the text-only Adaptive-RAG baseline on identical data — the
  governance and probe features carry real signal.
- The governance-scoped probe is the single load-bearing mechanism: removing it
  drops R@5 from 0.507 to **0.013** (*d* = −1.06) and abstention
  appropriateness by **0.698** (*d* = −1.52).
- A neural encoder beats the offline LSA default by **+0.022 R@5, p = 0.002** —
  significant at n ≈ 870, and *not* significant at n = 25, which is the clearest
  illustration of why the evaluation set had to grow.
- The learned router is the best of eight systems on every retrieval metric
  *and* the fastest: **R@5 0.548** against 0.537 for always-maximal retrieval,
  at 25% lower latency. The shipped rule-based router (0.524) is significantly
  worse than five of seven comparators — the architecture holds, the hand-tuned
  weights do not.
- Unspecialised retrieval statistics leak: with the route held fixed, deleting
  documents a principal cannot read changes their evidence ordering on **31.2%**
  of queries and the top result on **6.2%**. Scope-pure specialisation reduces
  both to **0%**. (An earlier unpinned version of this measurement reported
  "33.3% route changes"; that comparison also captured the deliberate
  `restricted_fraction` signal and should not be used — see `FINDINGS.md` §4.4.)
- **Zero ACL violations** across all eight systems and 965 queries.

**Known defects, found by the work and not yet fixed.** These are real and
should be picked up next:

1. **The evidence sufficiency gate is mis-calibrated at scale.** Disabling it
   *improves* abstention appropriateness by +0.105 (p < 0.0001). Its thresholds
   are absolute rerank scores tuned on 60 chunks and do not transfer to 7,082.
2. **Abstention on unanswerable questions is unreliable.** Four of five
   obviously-unsupported questions are answered anyway, because lexical overlap
   with real documents clears both the probe floor and the gate.
3. **Four governance mechanisms are inert** on the benchmark corpus: the
   restricted-scope signal, routing freshness penalty, routing conflict signal,
   and authority gate each produce exactly zero change on every metric.
4. **Freshness is measured but does not discriminate.** The generated
   freshness queries were originally unanswerable — all eight systems scored
   0.000 — which made `freshness_compliance` report a meaningless 1.000. The
   generator is fixed (0/15 → 15/15 retrieve their gold chunk, 0/15 lead with a
   superseded chunk) and guarded by
   `tests/test_evaluation.py::TestBenchmarkEvalSetIntegrity`. All eight systems
   now score 1.000, so the claim "freshness handling works" is supported and
   "ours is better than the baselines'" is not.

**Not attempted, and not claimable.** Inter-annotator agreement (§2b) and human
Likert scoring (§9c) both require a second person. Tooling can be built; the
judgements cannot be synthesised.

---

## 5. Running things

```bash
python -m pytest tests/ -q                      # 290 tests
python -m pytest tests/test_demonstration.py -s  # guided tour of the behaviours
python -m ahrag.evaluate                         # 8 systems on the demo corpus

# The experiment programme, in dependency order
python improvement_files/datasets/integrate_datasets.py
python improvement_files/corpus_sources/collect_public_corpus.py
python improvement_files/ml_router_training/train_router.py --integrated
python improvement_files/baseline_code/compare_baselines.py --integrated --by-type
python improvement_files/baseline_code/ablation_study.py --integrated
python improvement_files/evaluation_tools/compare_embeddings.py --integrated
python improvement_files/evaluation_tools/measure_specialisation.py --integrated

# Web
python -m uvicorn "ahrag.api.app:create_app" --factory --port 8000
streamlit run ahrag/ui/app.py
```

### Environment notes that will otherwise cost an hour

- **Pin the backends when testing.** `AHRAG_EMBEDDING_BACKEND=lsa` and
  `AHRAG_RERANKER=lexical`. With `auto`, installing the neural extra silently
  changes what the system does, and on Python 3.14 loading torch across
  session-scoped pytest fixtures segfaults the interpreter.
- **`xgboost` must be imported before `torch`.** Both bundle an OpenMP runtime
  and the wrong order aborts the process with no traceback.
  `ahrag/_openmp.py` handles this; do not remove its import from
  `ahrag/__init__.py`.
- **One sentence-transformer model per process.** Loading a second aborts
  similarly. Run multi-model comparisons as separate processes.
- Optional extras: `xgboost` + `scikit-learn` (learned router),
  `sentence-transformers` (neural encoder, needs `brew install libomp` for
  xgboost interop), `ragas` (LLM-judge metrics, needs `OPENAI_API_KEY`).

---

## 6. Key API surface

- **`AHRAGEngine.answer(query, user_id, history=None, write_audit=True)`** — the
  entry point. Handles ACL scoping, probing, feature extraction, routing,
  retrieval, evidence packing, conflict detection, sufficiency, generation,
  citation verification and audit. Returns `AnswerResult`.
- **`engine.acl.scope_for(user)`** — the authorised scope. Never bypass it;
  `assert_authorised` is a defence-in-depth re-check at every stage boundary and
  raising it indicates a pipeline bug, not user error.
- **`ahrag.eval.harness.build_seeded_engine(manifest, settings_overrides=...)`** —
  build an engine on a chosen corpus, with caching. Use this in scripts rather
  than constructing `AHRAGEngine()` directly, which silently defaults to the
  demo corpus.
- **`ahrag.eval.harness.load_and_validate(engine, eval_set)`** — load labels and
  verify they resolve against the loaded corpus.
- **`ahrag.routing.build_router(config, settings, backend)`** — `governance`,
  `learned`, `complexity`, or `adaptive-rag`.

### One caution for anyone writing an experiment script

Per-row keys from `ahrag.evaluate.run_item` are **not** the aggregate names from
`ahrag.eval.metrics.aggregate`. The row key is `abstention_appropriate` (a bool
per query); the aggregate is `abstention_appropriateness` (its mean). Reading
the aggregate name off a row yields `None` for every item, and the metric
silently reports "n/a" rather than failing.
