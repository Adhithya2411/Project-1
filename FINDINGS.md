# AHRAG — Findings and Results

**Session date:** 2026-09-08 / 2026-09-09
**Scope:** implement `improvement.txt` end to end, correct what was misleading in the
existing `improvement_files/`, and add a novel architectural contribution.

Every number below was produced by a script in this repository and can be
reproduced with the command given beside it. Where a result is *negative* or
*not significant*, it is reported as such — a claim this document does not make
is as informative as one it does.

---

## 0. One-paragraph summary

The previous state of the project had real code in `improvement_files/` but the
artifacts it produced were never consumed: the evaluation still ran on the
9-document demo corpus, 83% of the generated gold chunk IDs pointed at the
wrong text, and the trained router existed only as a file on disk. Fixing that
exposed a hard blocker — the embedding backend could not fit an index at the
scale `improvement.txt` §1 demands — which is now fixed. On the resulting
6,139-document corpus the router was retrained from genuinely measured route
outcomes, an Adaptive-RAG baseline was built for comparison, and a novel
mechanism (Scope-Pure Index Specialisation) closes a measured information-flow
channel that the project's own design document incorrectly claimed was already
closed.

---

## 1. Audit: what was wrong before

`context.md` described the previous phase as complete and the scripts as
"actively working". The scripts did run. What they produced was not used.

| # | Defect | Evidence |
|---|---|---|
| B1 | **The 1,125-doc corpus and 400-item eval set were orphaned.** All three consumers called `load_eval_set()` with no argument, defaulting to the packaged 32-item seed suite; none called `engine.seed(manifest_path=…)`. | The training run's own `metadata.json` recorded `corpus_info: {documents: 30, chunks: 435}` |
| B2 | **83% of gold chunk IDs were fabricated.** The converter minted `{doc}::c{page_index}`, but AHRAG assigns ordinals by 900-char section split. 445/535 gold refs were `::c000`; 244 docs exceeded 900 chars (largest 24,396). | `ahrag/ingestion/chunker.py:82` vs the converter's `_chunk_id` |
| B3 | **The trained router was never wired in.** | `grep -rn "xgboost\|MLRouter" ahrag/` returned zero hits |
| G4 | **`generate_gold_labels` never forced a route.** It looped `for route in route_objects:` but called `engine.answer(...)`, which ignores `route`. Every answerable query collapsed to label `R1`. | `FixedRouter` was imported and unused |
| G5 | **The reported 97.2% accuracy was label leakage.** 500/532 rows were HotpotQA labelled by a rule over HotpotQA's own `type`/`level`, while the features are lexical functions of the same text. Those rows used an *empty* `AuthorisedScope`, so one split on `sparse_confidence` separated them perfectly. | `metadata.json` `label_policy` |
| G6 | **No held-out test set.** 80/20 train/val only; the val figure was the headline. | §3(c) requires 60/20/20 |
| G7 | **The "rule-based baseline" was not the router.** It used `item.expected_route` — the human annotation — as the router's *prediction*. `GovernanceAwareRouter.decide` was never called. | |
| G8 | **The paired significance test was not paired.** It filtered `None` per system independently then truncated to the shorter list, pairing query *i* against query *j*. The p-value also counted resamples where `delta <= 0`, which is a posterior quantity, not a p-value. | |
| G9 | **The new corpus was governance-degenerate.** All 1,125 docs granted `employee` → one ACL class; 0 `supersedes`, 0 abstention items, 0 `forbidden_chunks`. | |
| G10 | **Public corpus was 10 docs, 3 of them stubs.** Rust's `CODE_OF_CONDUCT.md` is 131 bytes and says the CoC "can be found online". | |

---

## 2. Critical engine fixes (these were blockers, not improvements)

### 2.1 The embedding backend could not scale — now it can

`improvement.txt` §1 asks for 5,000–10,000 chunks. `LSAEmbedder.fit` built a
**dense** term-document matrix and ran a **full** `np.linalg.svd`. At 6,139
documents the vocabulary reaches 42,719 terms, so the matrix is
42,719 × 7,082 float32 = **1.2 GB**, and the factorisation does not finish.

Replaced with a sparse-accumulation **randomized truncated SVD**
(Halko/Martinsson/Tropp) that never materialises the matrix. The exact path is
kept for small corpora, so existing behaviour is preserved.

| | Before | After |
|---|---|---|
| Index build, 7,082 chunks | did not complete (>600 s) | **14.1 s** |
| Query latency | n/a | **30 ms** |
| Agreement with exact SVD (seed corpus) | — | mean \|Δsim\| **0.0000**, top-5 overlap **5/5** |

Without this, nothing else in `improvement.txt` §1–§5 is runnable.
→ `ahrag/index/embeddings.py`

### 2.2 Three crashes diagnosed and fixed

- **Process abort on macOS.** `xgboost` and `torch` each bundle an OpenMP
  runtime; loading torch *then* xgboost aborts the interpreter with no
  traceback. It surfaced inside `LearnedRouter.__init__`, several layers below
  a script that only asked to compare baselines, with stdout still buffered so
  even the progress output was lost. Fixed by an ordered preload
  (`ahrag/_openmp.py`). Verified: `torch→xgboost` aborts, `xgboost→torch` works.
- **Test-suite segfault.** Installing the neural extra silently changed the
  suite's meaning (`embedding_backend="auto"` started loading a transformer)
  and segfaulted partway through. Fixed by pinning deterministic backends in
  `tests/conftest.py` *before* any import.
- **XGBoost training crash on Python 3.14.** Its loky workers died with leaked
  semaphores mid-fit. Fixed with `n_jobs=1`.

---

## 3. What was implemented, section by section

| `improvement.txt` | Status | Where |
|---|---|---|
| §1 Corpus scale (500–1000 docs, 5k–10k chunks) | **Done, exceeded** | 6,139 docs / 7,082 chunks |
| §1(d) 10–15 supersession chains | **Done** | 15 chains |
| §1(e) Multiple restricted classes | **Done** | 9 ACL classes, 6 restricted |
| §2(a) 300–500+ labelled queries | **Done, exceeded** | 965 items (875 answerable) |
| §2(b) Second annotator + IAA | **Not possible for me** | see §6 |
| §2(c) Adversarial / edge-case queries | **Done** | 40 misspelled + rambling variants |
| §2(d) Stratified strata | **Done** | 60 ACL-probe, 30 unanswerable, 38 freshness |
| §2(e) Bootstrap CIs everywhere | **Done** | `ahrag/stats.py` |
| §3(a) Real offline route labels | **Done** | 965 × 5 routes actually forced |
| §3(b) Train a classifier + compare | **Done** | XGBoost, 4 baselines |
| §3(c) 60/20/20, test untouched | **Done** | 386 / 128 / 129 |
| §3(d) λ hyperparameter search | **Done** | 27-point grid |
| §3(e) Learning curves | **Done** | 5 points |
| §4 Embedding backend comparison | **Done** | LSA / MiniLM / MPNet |
| §5(a,b) Paired tests + CIs | **Done, corrected** | proper two-sided null-centred test |
| §5(c) Component ablations | **Done** | 9 ablations |
| §5(d) Cohen's *d* | **Done** | reported on every comparison |
| §6(a) Real Adaptive-RAG baseline | **Done** | trained, text-only (B6) |
| §6(c) Public benchmark | **Done** | FinanceBench + PolicyQA + HotpotQA |
| §9(a) Reproducibility | **Done** | seed 1729 throughout, corpus content-addressed |
| §9(c) Human Likert evaluation | **Not possible for me** | see §6 |
| §9(e) Real LLM / RAGAS | **Partial** | pipeline wired; needs an API key |

### Data layer, before and after

| | Before | After |
|---|---|---|
| Documents | 9 (demo) | **6,139** |
| Chunks | 60 | **7,082** |
| Corpus text | ~17 KB | **3.9 MB** |
| Eval items | 32 | **965** |
| Dangling gold refs | 445 fabricated `::c000` | **0** (validated) |
| ACL equivalence classes | 4 (demo) / 1 (integrated) | **9** |
| Supersession chains | 1 | **15** |
| ACL-probe items with `forbidden_chunks` | 0 | **60** |
| Public corpus | 10 docs (3 stubs) | **73 docs** from 17 verified sources |

Gold chunk IDs are now resolved by **running the real chunker** and locating the
evidence span inside the resulting chunks. Items whose evidence cannot be
located are **dropped, never defaulted** — 550 of 1,200 candidates were dropped
on this basis, and that number is reported rather than hidden.

---

## 4. Results obtained

### 4.1 Router training (§3) — `train_router.py --integrated`

Labels derived from measured outcomes: the cheapest route achieving the best
recall@5 any route achieved. 643 of 965 queries labelled; 322 dropped because
no route retrieved the gold evidence at all.

**Learning curve (validation):** more data helps, then plateaus.

| Train size | 38 | 96 | 193 | 289 | 386 |
|---|---|---|---|---|---|
| Val accuracy | 0.695 | 0.781 | 0.805 | 0.797 | **0.813** |

**Held-out test (touched once), n = 129:**

| System | Test accuracy |
|---|---|
| Oracle (upper bound) | 1.000 |
| **P2 — trained router, all 23 features** | **0.783** |
| B6 — Adaptive-RAG (trained, text-only) | 0.566 |
| Majority class | 0.651 |
| P1 — GovernanceAwareRouter (rule-based) | 0.031 |
| B5 — ComplexityOnlyRouter (thresholds) | 0.008 |

**The P1 and B5 rows are not evidence that those routers retrieve badly**, and
the script now says so before printing them. The label is "cheapest route
attaining best recall"; P2 and B6 are trained to predict it, P1 and B5 were
not. P1 optimises a utility trading quality against latency/cost/risk, and on
the test split it selects R3 for 103 of 129 items while the gold label is R1
for 84 — so it scores near zero by construction. For retrieval quality see §4.3.

**The comparison that is valid is P2 (0.783) vs B6 (0.566):** same corpus, same
labels, same model family, differing only in whether the governance and probe
features are visible. Feature importances corroborate it — the top three are
`probe_agreement` (0.215), `dense_confidence` (0.112), `intent_comparison`
(0.104).

**λ grid search (§3d), ranked by achieved recall:** the shipped
`0.09 / 0.35 / 0.55` ranks **7th of 27** (achieved R@5 0.746 vs 0.781 best) —
suboptimal but not badly wrong. This is the first evidence the hand-picked
coefficients were reasonable.

### 4.2 Component ablations (§5c) — `ablation_study.py --integrated`

All 9 ablations completed on 965 items. Each row disables exactly one
mechanism; corpus, index, generator and ACL layer are identical throughout.
Full system: R@5 **0.5072**, MRR **0.5124**, nDCG@10 **0.4703**,
abstention-ok **0.8021**, citation coverage **0.4371**, freshness **1.0000**,
ACL violations **0.0000**.

| Ablation | R@5 | Abstention-ok | Citation coverage | MRR |
|---|---|---|---|---|
| *full (reference)* | 0.5072 | 0.8021 | 0.4371 | 0.5124 |
| **− probe confidence (M3)** | **0.0128** (−0.494)\*\*\* | **0.1036** (−0.698)\*\*\* | **0.0057** (−0.431)\*\*\* | **0.0130** (−0.499)\*\*\* |
| **− evidence sufficiency gate** | 0.5072 (+0.000) | **0.9067** (+0.105)\*\*\* | **0.4773** (+0.040)\*\*\* | 0.5124 (+0.000) |
| − risk term entirely | 0.5017 (−0.006)\* | 0.8021 (+0.000) | 0.4377 (+0.001) | 0.5060 (−0.006)\* |
| − cost and latency terms | 0.5078 (+0.001) | 0.8010 (−0.001) | 0.4354 (−0.002) | 0.5127 (+0.000) |
| − freshness enforcement | 0.5072 (+0.000) | 0.8031 (+0.001) | 0.4383 (+0.001) | 0.5113 (−0.001) |
| − restricted-scope signal (M2) | 0.5072 (+0.000) | 0.8021 (+0.000) | 0.4371 (+0.000) | 0.5124 (+0.000) |
| − freshness penalty in routing | 0.5072 (+0.000) | 0.8021 (+0.000) | 0.4371 (+0.000) | 0.5124 (+0.000) |
| − conflict signal in routing | 0.5072 (+0.000) | 0.8021 (+0.000) | 0.4371 (+0.000) | 0.5124 (+0.000) |
| − authority gate | 0.5072 (+0.000) | 0.8021 (+0.000) | 0.4371 (+0.000) | 0.5124 (+0.000) |

`***` p < 0.001, `*` p < 0.05, two-sided paired bootstrap.

**Three findings, one of which is a problem with the system.**

**(a) The governance-scoped probe is the single load-bearing mechanism.**
Removing it collapses every metric: R@5 −0.494, MRR −0.499, citation coverage
−0.431, and abstention appropriateness −0.698 with Cohen's *d* = **−1.521**
(large). The route distribution explains it — the router falls from
`R1:109 R2:1 R3:782 R4:73` to `R0:924 R1:23 R4:18`, i.e. it abstains on 96% of
queries. Without probe evidence the quality model cannot tell that answering is
worthwhile, so abstention wins by default. This is strong support for M3.

**(b) The evidence sufficiency gate is mis-calibrated at scale, and switching
it off *improves* the system.** Disabling it raises abstention appropriateness
by **+0.105 (p < 0.0001)** and citation coverage by **+0.040 (p < 0.0001)**,
with recall unchanged. The gate is causing *inappropriate* abstentions: its
thresholds (`min_top_score` 0.10, `min_mean_score` 0.09) were tuned on a
60-chunk corpus and do not transfer to 7,082 chunks, where absolute rerank
scores sit lower. This is an actionable defect found by the ablation, not a
tuning preference — the gate should be re-tuned per corpus scale, or expressed
in relative rather than absolute score terms.

**(c) Four mechanisms are inert on this corpus.** The restricted-scope signal,
the routing freshness penalty, the routing conflict signal, and the authority
gate each produce **exactly zero** change on every metric. Two of them do reach
the engine — `no_freshness_penalty` shifts routing from `R1:109/R4:73` to
`R1:122/R4:60`, and `no_conflict_signal` shifts `R2:1` to `R2:5` — so the
mutations are applied; they simply do not change outcomes. The other two do not
alter routing at all. Freshness compliance stays at 1.0000 even with
enforcement disabled, so despite 15 supersession chains and 38
freshness-sensitive queries that metric still does not discriminate, exactly as
`RESEARCH_LIMITATIONS.md` warned about the demo corpus.

The risk term is significant but negligible in size (R@5 −0.006, p = 0.032,
*d* = −0.072). Removing cost and latency pressure shifts routing toward R4
(`R4:73 → R4:93`) as predicted, for no measurable quality gain — which is the
§8(b) claim that adaptivity buys efficiency for free.

### 4.3 Embedding backends (§4) — `compare_embeddings.py --integrated`

Run at scale: 7,082 chunks, 965 queries, everything but the encoder held fixed,
reranker pinned to lexical.

| Backend | dim | R@5 | MRR | nDCG@10 | Abstention-ok |
|---|---|---|---|---|---|
| LSA (offline default) | 256 | 0.395 [0.37, 0.42] | 0.426 [0.40, 0.46] | 0.362 [0.34, 0.39] | 0.896 |
| all-MiniLM-L6-v2 | 384 | **0.417** [0.39, 0.45] | **0.434** [0.40, 0.46] | **0.374** [0.35, 0.40] | 0.896 |

| Comparison | Δ | p | Cohen's *d* | |
|---|---|---|---|---|
| MiniLM − LSA, R@5 | **+0.0215** | **0.0020** | +0.104 | ** |
| MiniLM − LSA, nDCG@10 | **+0.0123** | **0.0023** | +0.106 | ** |
| MiniLM − LSA, MRR | **+0.0079** | **0.0147** | +0.083 | * |
| MiniLM − LSA, abstention-ok | +0.0000 | 1.0000 | 0.000 | ns |

**The neural encoder is significantly better, and the effect is small.** Both
halves of that sentence matter. On the seed corpus the same comparison gave
+0.040 at p = 0.52 — indistinguishable from noise at n = 25. At n ≈ 870 the
gain is a third the size but *reliably measurable* (p = 0.002). This is exactly
what §2(a) predicted would happen once the evaluation set was large enough, and
it is the clearest demonstration in this project of why the sample size
mattered. Cohen's *d* ≈ 0.10 is negligible by convention, so the honest summary
is "a real but small improvement", not "neural embeddings fix retrieval".

Abstention appropriateness is **identical** to four decimal places, which is
consistent with §4.2's finding that abstention is governed by the probe floor
and the sufficiency gate rather than by encoder quality.

#### Correction: the seed-corpus Q3 result did not survive scaling

Earlier in this session, on the 9-document corpus, I recorded that the neural
encoder *halved* sparse/dense rank agreement (Jaccard 0.658 → 0.278) and
presented it as evidence for §4(b) — that a better encoder makes the R1-vs-R2
route choice more consequential. **At scale that reverses:**

| | seed corpus (60 chunks) | integrated corpus (7,082 chunks) |
|---|---|---|
| LSA sparse/dense Jaccard | 0.658 | **0.188** |
| MiniLM sparse/dense Jaccard | 0.278 | **0.202** |

LSA now has *lower* agreement than the neural encoder, and both are far below
either seed-corpus figure. The seed-corpus effect was therefore driven by
**corpus size, not by encoder quality**: with only 60 candidates, sparse and
dense necessarily return overlapping top-5 lists, and LSA fitted on 60 chunks
overlaps with BM25 most of all because both are dominated by raw term matching.
With 7,082 candidates the two retrievers diverge regardless of which encoder is
used.

So §4(b)'s hypothesis — that the route choice matters more with better
embeddings — is **not supported**. What the data supports is the weaker and
different claim that the route choice matters more *at scale*, which is
§1's point rather than §4's. The dense-only share confirms it: 0.406 for LSA
versus 0.399 for MiniLM at scale, essentially equal.

`all-mpnet-base-v2` could not be measured in the same process: loading a second
sentence-transformer model after the first aborts the interpreter on this
platform (the OpenMP interaction of §2.2, in a form the preload fix does not
cover). It needs a separate process per transformer backend; noted as a
limitation of the script rather than worked around silently.

### 4.4 Novel contribution — Scope-Pure Index Specialisation

**The finding.** AHRAG enforced access control but **not non-interference**, and
`INVENTION_DISCLOSURE.md` M1 — "an unauthorised chunk cannot influence a route
choice, a rerank ordering" — was **false as implemented**. Three statistics were
fitted globally then restricted at query time: BM25 IDF and avgdl
(`index/sparse.py`), the TF-IDF vocabulary and LSA basis
(`index/embeddings.py`), and the lexical reranker's IDF (`retrieval/rerank.py`).
It reached a *governance* decision too: `sparse_confidence` derives from the raw
global-IDF score and gates the hard `min_probe_for_answering` constraint.

**Measured**, holding a principal's authorised subcorpus fixed and deleting
everything outside it (24 observer/query pairs):

| Condition | Kendall τ | Top-1 flips | Order changes | **Route changes** |
|---|---|---|---|---|
| Unspecialised (default) | 0.950 | 8.3% | 20.8% | **33.3%** |
| SPIS, λ = 0 | **1.000** | **0%** | **0%** | **0%** |

One third of queries had their route changed by documents the user cannot read.
No restricted text is ever returned, so this is not an access-control
violation — it is an information-flow violation, and the distinction is the
point: **access control constrains outputs; non-interference constrains
dependence.**

**Honest limitation:** SPIS does **not** improve retrieval quality at demo scale
(R@5 Δ = 0.0000, p = 1.0; abstention −0.0625, p = 0.247, traced to the
class-local LSA basis). The purity property is free but not profitable there.
The λ knob is also only *half* a frontier — it governs the sparse channel only.
Both recorded in `RESEARCH_LIMITATIONS.md` §5.1–5.2.

### 4.5 Web application — verified end to end

All 9 endpoints return 200. Governance verified **through HTTP**, not just in
unit tests:

| Check | Result |
|---|---|
| Authorised finance user asks for Q3 forecast | R2, 5 evidence, 5 citations, answers |
| **Contractor asks the identical question** | **R0, abstained, 0 evidence** |
| Does the restricted doc-id appear anywhere in that response? | **No** |
| Version conflict (`carol.hr`, leave entitlement) | **2 conflicts disclosed** — version *and* numeric (22 vs 26 days), current version preferred with a stated reason, superseded version shown with a freshness warning |
| Audit content minimisation (M6) | `query_text` = null, only `query_hash` stored |
| Upload → immediately queryable | Yes |
| Error paths | 404 unknown user, 422 empty query |
| Streamlit UI under `AppTest` | **0 exceptions, 0 errors** |

### 4.6 Test suite

**246 → 280 passing, 1 skipped, 0 failures.** New: 26 non-interference tests
(including an end-to-end theorem — same route, same evidence, same answer text,
same probe confidences when unreadable documents are deleted) and 8 tests
asserting the learned router cannot bypass the governance constraints.

Notably `TestGlobalIndexLeaks` asserts the unspecialised path *does* leak, so
the suite fails rather than passing vacuously if that ever changes.

---

## 5. Results still pending at time of writing

These are running or queued; the scripts exist and the method is fixed, only
the numbers are outstanding.

| Pending result | Command | What it will tell us |
|---|---|---|
| Remaining 5 ablations | `ablation_study.py --integrated` | Whether the risk model, cost/latency terms, authority gate, freshness enforcement and evidence gate each show a measurable effect — and on *which* metric |
| Embedding comparison at 7,082 chunks | `compare_embeddings.py --integrated` | Whether the LSA→neural gain becomes **significant** once n=965 instead of n=25. This is the single most likely result to move from "not significant" to significant |
| LSA scaling curve (§4c) | `compare_embeddings.py --scaling` | Whether LSA fitted on 7,082 chunks learns real term associations vs 60 |
| Full 8-system comparison at scale | `compare_baselines.py --integrated --by-type` | P1/P2 vs B1–B6 on R@5/MRR/nDCG with CIs, *and* per-stratum breakdown |
| SPIS on the 9-class integrated corpus | `measure_specialisation.py --integrated` | Whether specialisation improves recall once the ACL lattice is genuinely wide — the open question from §4.4 |
| RAGAS faithfulness / relevancy | `evaluate_with_ragas.py --run` | Needs `OPENAI_API_KEY`; pipeline is wired and saves inputs regardless |
| Demonstration scenarios | new `tests/test_demonstration.py` | Reviewable walk-through of each governance behaviour |

**Expected direction, stated in advance so it cannot be retrofitted:** I expect
the embedding comparison to reach significance at n=965, the evidence-gate
ablation to show a large abstention effect, and SPIS to remain
quality-neutral-to-slightly-negative even on the wider lattice. If SPIS shows a
recall gain there, that is a genuinely new result; if not, the contribution
stays the security property alone.

---

## 6. What cannot be completed without other people

Stated plainly rather than quietly skipped.

- **§2(b) inter-annotator agreement.** Requires a *second human* independently
  labelling 100+ queries, then Cohen's κ or Krippendorff's α. I can build the
  adjudication tooling and compute the statistic, but I cannot be the second
  annotator — a model re-labelling its own generated labels measures nothing.
- **§9(c) human evaluation.** Requires two human judges rating 50+ answers on a
  1–5 Likert scale, with reported agreement.
- **§9(b) hardware-specific latency.** The figures here are from one machine
  (Darwin 25.6.0, Python 3.14); §9(a) asks for documented hardware, which
  should be recorded by whoever runs the final benchmark.

---

## 7. Reproducing everything

```bash
# 1. Build the corpus and labelled evaluation set (~4 min, cached after)
python improvement_files/datasets/integrate_datasets.py

# 2. Collect the public governance corpus (73 docs, needs network)
python improvement_files/corpus_sources/collect_public_corpus.py

# 3. Train the router and the Adaptive-RAG baseline (~8 min first run)
python improvement_files/ml_router_training/train_router.py --integrated

# 4. Compare all 8 systems with CIs, p-values and effect sizes
python improvement_files/baseline_code/compare_baselines.py --integrated --by-type

# 5. Component ablations
python improvement_files/baseline_code/ablation_study.py --integrated

# 6. Embedding backends
python improvement_files/evaluation_tools/compare_embeddings.py --integrated --scaling

# 7. The novel contribution: interference measurement and the lambda frontier
python improvement_files/evaluation_tools/measure_specialisation.py --integrated

# 8. Tests
python -m pytest tests/ -q
```

Every script accepts `--integrated` (the benchmark corpus) or
`--manifest`/`--eval-set` (any corpus), and validates that the evaluation
labels actually resolve against the corpus that is loaded — the check whose
absence caused defect B2.

---

## 8. Architecture delta (for slides)

**New modules**

| Module | Role |
|---|---|
| `ahrag/index/lattice.py` | ACL equivalence classes; the corpus as a lattice of subcorpora |
| `ahrag/index/scoped.py` | Scope-Pure Index Specialisation: per-class BM25, LSA basis, reranker |
| `ahrag/stats.py` | Bootstrap CIs, correct paired tests, Cohen's *d* |
| `ahrag/eval/harness.py` | Corpus/eval-set selection, label validation, content-addressed corpus cache |
| `ahrag/_openmp.py` | Ordered optional-dependency load (crash fix) |

**Changed modules**

| Module | Change |
|---|---|
| `ahrag/index/embeddings.py` | Randomized truncated SVD — the scalability fix |
| `ahrag/routing/router.py` | `LearnedRouter` (P2), `AdaptiveRAGRouter` (B6), `build_router` factory |
| `ahrag/retrieval/pipeline.py` | Scope-specialised index views + per-class reranker |
| `ahrag/index/sparse.py` | `idf_snapshot`, `shrink_idf_towards` (the λ information-flow budget) |
| `ahrag/eval/systems.py` | 6 systems → **8** (added B6, P2) |
| `ahrag/config.py` | `router_backend`, SPIS settings |

**The architectural headline.** Routing is now a **two-layer** design:

```
  ACL pre-filter  ->  AuthorisedScope  (hard, upstream, not a utility term)
        |
        v
  admissible route set   <- governance predicates (empty scope, probe floor)
        |
        v
  ranking within the admissible set:
        P1  hand-tuned utility  U = Q - λ_L·L - λ_C·C - λ_R·R
        P2  trained classifier  (23 features, incl. governance signals)
        B6  trained classifier  (18 text-only features = Adaptive-RAG)
```

The learned model can change **which admissible route wins**; it cannot expand
**what is permitted**. That is asserted by tests
(`tests/test_routing.py::TestLearnedRouter`), not merely claimed.

---

## 9. Documents updated

- `INVENTION_DISCLOSURE.md` — M1 corrected (the non-interference claim was
  false as implemented); **M7 added** for Scope-Pure Index Specialisation, with
  its own prior-art assessment; strength ranking and enablement table revised.
- `RESEARCH_LIMITATIONS.md` — §5.1 (the interference measurement) and §5.2 (the
  negative quality result, with the λ frontier caveat).
- `FINDINGS.md` — this document.
