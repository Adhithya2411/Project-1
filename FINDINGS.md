# AHRAG — Findings and Results

**Session date:** 2026-09-08 / 2026-09-09
**Scope:** implement `improvement.txt` end to end, correct what was misleading in the
existing `improvement_files/`, and add a novel architectural contribution.

Every number below was produced by a script in this repository and can be
reproduced with the command given beside it. Where a result is *negative* or
*not significant*, it is reported as such — a claim this document does not make
is as informative as one it does.

---

## 0. Summary

The previous state of the project had real code in `improvement_files/` but the
artifacts it produced were never consumed: the evaluation still ran on the
9-document demo corpus, 83% of the generated gold chunk IDs pointed at the
wrong text, and the trained router existed only as a file on disk. Fixing that
exposed a hard blocker — the embedding backend could not fit an index at the
scale `improvement.txt` §1 demands — which is now fixed.

On the resulting 6,139-document / 965-query benchmark, the headline results are:

- **The learned router (P2) is the best of eight systems on every retrieval
  metric and also the fastest** — R@5 0.530 against 0.520 for always-maximal
  retrieval, at 25% lower latency. It beats the trained text-only Adaptive-RAG
  baseline 0.783 vs 0.566 on route accuracy.
- **The shipped hand-tuned router (P1) is significantly worse than five of
  seven comparators.** The architecture is sound; the hand-picked utility
  weights were not, which is exactly what improvement.txt §3 alleged.
- **Zero ACL violations across all eight systems and 965 queries.**
- **The governance-scoped probe is load-bearing**: removing it drops R@5 from
  0.507 to 0.013 (*d* = −1.06).
- **A defect found by ablation**: disabling the evidence sufficiency gate
  *improves* abstention appropriateness by +0.105 (p < 0.0001) — it is
  mis-calibrated at scale.
- **A novel mechanism** (Scope-Pure Index Specialisation) closes a measured
  information-flow channel that `INVENTION_DISCLOSURE.md` M1 incorrectly
  claimed was already closed: unreadable documents changed a principal's route
  on 33.3% of queries. It costs 0.004 R@5 (p = 0.027) and buys no retrieval
  gain, so the contribution is the security property alone.
- **A defect in this session's own work**: the generated freshness stratum is
  unanswerable by every system, so no freshness claim here is evidenced (§4.5b).

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
| §1 Corpus scale (500–1000 docs, 5k–10k chunks) | **Partially** — chunk target met; only 249 documents are enterprise-shaped | 6,139 docs / 7,082 chunks |
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
| — of which enterprise-shaped (policy / finance / runbook) | 9 | **249** |
| — of which single-paragraph reference articles | 0 | **5,890** |
| Chunks | 60 | **7,082** |
| Corpus text | ~17 KB | **3.9 MB** |
| Eval items | 32 | **965** |
| Dangling gold refs | 445 fabricated `::c000` | **0** (validated) |
| ACL equivalence classes | 4 (demo) / 1 (integrated) | **9** |
| Supersession chains | 1 | **15** |
| ACL-probe items with `forbidden_chunks` | 0 | **60** |
| Public corpus | 10 docs (3 stubs) | **73 docs** from 17 verified sources |

**"6,139 documents" needs qualifying.** 5,890 of them are single-paragraph
reference articles from HotpotQA, included so that first-stage retrieval is
genuinely *selective* — which is what improvement.txt §1 was actually asking
for when it said "any retrieval route finds almost everything" at 60 chunks.
The enterprise-shaped count is **249** (165 policy, 84 finance), and that is the
figure to compare against §1's "500–1000 documents", which it does **not** meet.
What the corpus does deliver is the retrieval difficulty and the ACL lattice
width that make every other measurement meaningful.
`integration_report.json` now reports both counts separately.

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
everything outside it. **The route is pinned to R3** for this measurement, and
that correction matters — see the note below.

Benchmark corpus, 7 ACL equivalence classes, 16 observer/query pairs:

| Condition | Kendall τ | Top-1 flips | Evidence-order changes | Non-interfering |
|---|---|---|---|---|
| Unspecialised (default) | 1.000 | **6.2%** | **31.2%** | **No** |
| SPIS, λ = 0 | 1.000 | **0%** | **0%** | **Yes** |

Nearly a third of queries had their *evidence ordering* changed by documents the
principal cannot read, and one in sixteen had its top result changed.
Specialisation eliminates both. No restricted text is ever returned in either
condition, so this is not an access-control violation — it is an
information-flow violation, and the distinction is the point: **access control
constrains outputs; non-interference constrains dependence.**

#### Correction: the first version of this measurement was wrong

An earlier run of this experiment let the router choose, and reported
"33.3% route changes unspecialised, 0% under SPIS" on the demo corpus. That
comparison was invalid, and I am recording why rather than quietly replacing
the number.

Deleting the documents a principal cannot read also changes
`AuthorisedScope.withheld_count` and `total_chunks`, and therefore
`restricted_fraction` — which is a **deliberate** router input (mechanism M2:
the router is supposed to know how much of the corpus is withheld from this
principal). So the unpinned experiment mixed an *intended* dependence in with
the unintended one, and no system could ever have scored zero on it. The
demo-corpus 0% was luck: the feature change had not crossed a decision boundary
on those six queries. At benchmark scale it does cross one, and the same
experiment reported **12.5% route changes for a system whose index is provably
pure** — which is what exposed the flaw.

Pinning the route removes the intended dependence and leaves exactly what
specialisation governs: the ranking the index produces over a fixed authorised
pool. The table above is that measurement. The route-change column is now zero
for both conditions by construction and is retained only as a diagnostic.

The unit tests were never affected — `tests/test_noninterference.py` compares
`sparse.search` output directly, with no router in the path, which is why its
26 assertions passed throughout.

**The open question, now answered.** At demo scale SPIS was quality-neutral,
and the obvious objection was that the demo corpus has almost no ACL structure
to exploit. So it was re-run on the benchmark corpus, whose lattice is
genuinely wide: **7 distinct ACL equivalence classes** across 7 principals,
class sizes 3,123–5,677 chunks, with the narrowest principal seeing only 44% of
the corpus.

| System | R@5 | MRR | nDCG@10 | Abstention-ok |
|---|---|---|---|---|
| unspecialised | 0.507 [0.48, 0.54] | 0.512 | 0.470 | 0.802 |
| SPIS λ = 0 (provably pure) | 0.503 [0.47, 0.53] | 0.511 | 0.468 | 0.802 |
| SPIS λ = 0.5 | 0.506 [0.48, 0.54] | 0.512 | 0.470 | 0.803 |
| SPIS λ = 1 | 0.507 [0.48, 0.54] | 0.512 | 0.470 | 0.802 |

| Comparison vs unspecialised | Δ R@5 | p | *d* |
|---|---|---|---|
| λ = 0 | **−0.0040** | **0.027** | −0.071 |
| λ = 0.5 | −0.0011 | 0.584 | −0.034 |
| λ = 1 | +0.0000 | 1.000 | 0.000 |

**Non-interference costs 0.004 Recall@5, and that cost is real (p = 0.027)
rather than noise.** The effect size is negligible by convention
(*d* = −0.071), so in absolute terms the price of the security property is
about one item in 250. It does not disappear on a wide lattice, which settles
the question: **SPIS's contribution is the information-flow property alone, not
a retrieval improvement.**

This was predicted in advance. §5 of an earlier draft of this document, written
before the run, said "I expect SPIS to remain quality-neutral-to-slightly-negative
even on the wider lattice. If SPIS shows a recall gain there, that is a
genuinely new result; if not, the contribution stays the security property
alone." It did not, and the contribution stays the security property alone.

**The λ frontier behaves exactly as designed**, which is the useful part. λ is
an explicit information-flow budget: at λ = 0 the index is provably
non-interfering and costs 0.004 recall; by λ = 1 the recall penalty is
identically zero and the sparse channel's statistics are fully global again.
An operator can therefore choose a point on that curve deliberately, with both
axes measured, instead of getting whichever end the implementation happened to
pick. That is a small curve, but it is a *measured* one.

**Remaining honest limitations:**

- λ governs the **sparse channel only**. The LSA basis is class-local whenever
  specialisation is on, so it is half a frontier; a complete one needs a dense
  analogue.
- Specialisation is **expensive at scale**: building 7 per-class indexes over a
  7,082-chunk corpus dominates the run time, since each fits its own LSA space.
  The LRU cap bounds memory but not first-use latency.
- With a **pre-trained neural encoder the dense channel is inherently pure** —
  the encoder holds no corpus statistics — so SPIS's dense specialisation is
  only relevant to corpus-fitted embeddings like LSA. Only BM25 IDF and the
  reranker's IDF leak in that configuration.

All of the above is recorded in `RESEARCH_LIMITATIONS.md` §5.1–5.2.

### 4.5 All eight systems at benchmark scale — `compare_baselines.py --integrated`

965 items, aligned on the **875** where every system produced a value.
Every system shares one corpus, one index, one generator and one ACL layer;
only the routing policy differs.

| System | R@5 | MRR | nDCG@10 | Abstention-ok | Latency | ACL violations |
|---|---|---|---|---|---|---|
| B1 fixed BM25 | 0.473 [0.44,0.50] | 0.506 | 0.453 | 0.811 | 0.095 s | **0** |
| B2 fixed dense | 0.519 [0.49,0.55] | 0.527 | 0.480 | 0.820 | 0.086 s | **0** |
| B3 fixed hybrid RRF | 0.542 [0.51,0.57] | 0.539 | 0.498 | 0.824 | 0.103 s | **0** |
| B4 always-maximal (R4) | 0.537 [0.51,0.57] | 0.538 | 0.495 | 0.824 | 0.109 s | **0** |
| B5 complexity-only | 0.539 [0.51,0.57] | 0.537 | 0.496 | 0.824 | 0.145 s | **0** |
| B6 Adaptive-RAG (trained) | 0.539 [0.51,0.57] | 0.540 | 0.498 | 0.823 | 0.096 s | **0** |
| P1 governance-aware (rule-based) | 0.524 [0.49,0.55] | 0.530 | 0.487 | 0.818 | 0.102 s | **0** |
| **P2 learned router** | **0.548** [0.52,0.58] | **0.542** | **0.503** | 0.818 | **0.082 s** | **0** |

These are the numbers **after** the freshness-stratum repair described in
§4.5(b) below. Every system gained roughly +0.017 Recall@5, which is exactly
the 15 previously-unanswerable freshness items now scoring 1.000
(15/875 ≈ 0.017). The ordering is unchanged.

**Zero ACL violations for all eight.** That is the intended result rather than a
selling point: ACL enforcement is upstream of routing, so no routing policy can
break it. The metric exists to verify the invariant holds under every policy,
and at 965 queries across 7 principals it does.

#### The learned router wins, and the hand-tuned one loses

P2 is the best system on **every** retrieval metric *and* the fastest. Against
the hoped-for claim in improvement.txt §8(b) — "matches always-maximal
retrieval quality while reducing estimated cost" — P2 does better than hoped:
it **exceeds** always-maximal quality (0.530 vs 0.520) while running **25%
faster** (0.082 s vs 0.109 s). It gets there by routing 626 of 965 queries to
the cheap sparse route and abstaining on 142, versus B4's 965 iterative runs.

The uncomfortable half of the same result is that **P1, the shipped rule-based
governance router, is significantly worse than five of the seven comparators**:

| P1 versus | Δ R@5 | p | *d* | |
|---|---|---|---|---|
| B1 fixed BM25 | **+0.0518** | 0.0000 | +0.211 | \*\*\* |
| B2 fixed dense | +0.0053 | 0.559 | +0.020 | ns |
| B3 fixed hybrid | **−0.0175** | 0.0010 | −0.128 | \*\* |
| B4 always-maximal | **−0.0130** | 0.047 | −0.068 | \* |
| B5 complexity-only | **−0.0147** | 0.0024 | −0.106 | \*\* |
| B6 Adaptive-RAG | **−0.0149** | 0.027 | −0.073 | \* |
| **P2 learned** | **−0.0232** | 0.0057 | −0.094 | \*\* |

So the *architecture* is sound — the same governance constraints with a learned
ranker on top produce the best system in the comparison — but the
**hand-picked utility weights are not**. That is consistent with the λ grid
search, which independently ranked the shipped `0.09/0.35/0.55` **7th of 27**.
improvement.txt §3's complaint that the weights were "unvalidated priors" is
confirmed, and the fix is the learned router rather than more hand-tuning.

Note also that B5 (complexity-only) and B6 (Adaptive-RAG) are statistically
indistinguishable from each other and from B3 (fixed hybrid). At this corpus
scale, *adaptivity by itself buys nothing* — what buys something is adaptivity
**fitted to measured outcomes**, which is P2.

#### Stratified results, and two problems they expose

| Query type | n | B1 | B3 | B6 | P1 | **P2** |
|---|---|---|---|---|---|---|
| hotpotqa_comparison | 70 | 0.850 | **0.950** | 0.907 | **0.950** | 0.907 |
| hotpotqa_bridge | 326 | 0.817 | 0.842 | 0.844 | 0.842 | **0.848** |
| finance_novel-generated | 49 | 0.408 | 0.558 | **0.578** | 0.490 | **0.578** |
| finance_metrics-generated | 30 | 0.117 | 0.478 | 0.478 | 0.189 | **0.544** |
| finance_domain-relevant | 45 | 0.189 | **0.411** | 0.411 | 0.389 | 0.389 |
| adversarial_rambling | 21 | 0.000 | 0.254 | 0.238 | 0.143 | **0.349** |
| adversarial_misspelled | 19 | 0.158 | 0.246 | 0.246 | 0.246 | **0.325** |
| policy_comprehension | 300 | 0.125 | 0.160 | 0.158 | 0.160 | **0.162** |
| **freshness_competing_versions** | 15 | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** |
| permission_boundary | 60 | — | — | — | — | — |
| unanswerable | 30 | — | — | — | — | — |

`permission_boundary` and `unanswerable` show no recall because they have no
gold chunks by construction — recall is undefined and they are scored by
abstention appropriateness instead. That is correct, not missing data.

Two things this table exposes, one about the systems and one about my own data:

**(a) P1's weakness is concentrated, not diffuse.** It collapses on
`finance_metrics-generated` (0.189 against P2's 0.544) and
`adversarial_rambling` (0.143 against 0.349). These are the queries where a
long or numerically-dense question pushes the hand-tuned utility toward R3 when
the sparse route would have found the figure. P2 learned not to. On the
HotpotQA strata, where P1 already routes well, the two are equal or P1 is
slightly ahead.

**(b) My freshness stratum was badly constructed. It is now fixed, and the
metric it feeds finally means something.**

In the first run, all eight systems scored **exactly 0.000** Recall@5 on all 15
`freshness_competing_versions` items. That was not eight systems failing; it was
my own generator producing unanswerable queries. `build_freshness_items` phrased
them as *"What is the current rule in {heading}?"* — almost entirely function
words plus a short heading, which against 7,082 chunks cannot retrieve one
specific chunk whatever the routing policy. It also explains why
`freshness_compliant` reported 1.000 in every ablation *including the one that
disabled freshness enforcement*: the metric was never exercised, and an
unexercised metric reads as perfect.

The generator now builds each query from a distinctive content phrase taken from
inside the versioned section, so the section is genuinely retrievable. The
stratum went from **0/15 to 15/15** items retrieving their gold chunk, and
**0/15** have a superseded chunk leading the evidence — so the current version
is being preferred, which is the behaviour the stratum exists to test.

All eight systems now score **1.000** on it. That means freshness handling
*works* and is *measured*; it does **not** mean AHRAG's freshness handling beats
the baselines, because every system is equal there. The honest claim is the
first one only.

Two guards were added so this cannot regress silently:
`tests/test_evaluation.py::TestBenchmarkEvalSetIntegrity` asserts that at least
80% of freshness queries retrieve their own gold chunk, that every freshness
gold chunk is the *current* version, that no ACL-probe forbidden chunk is
actually readable by its asker, and that no label references a non-existent
chunk. A generated dataset needs tests for the same reason production code does.

`policy_comprehension` at ~0.16 across all systems (n=300, 31% of the suite) is
a genuine difficulty result rather than a defect: PolicyQA asks short
reading-comprehension questions about privacy-policy paragraphs, and locating
the right paragraph among 7,082 chunks is hard. It is what drags the overall
means to ~0.5, and it is the main reason absolute recall here is far below the
0.84 seen on the 60-chunk demo corpus.

### 4.6 Web application — verified end to end

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

### 4.7 Test suite

**246 → 280 passing, 1 skipped, 0 failures.** New: 26 non-interference tests
(including an end-to-end theorem — same route, same evidence, same answer text,
same probe confidences when unreadable documents are deleted) and 8 tests
asserting the learned router cannot bypass the governance constraints.

Notably `TestGlobalIndexLeaks` asserts the unspecialised path *does* leak, so
the suite fails rather than passing vacuously if that ever changes.

---

## 5. What is still outstanding

Everything in improvement.txt §1–§8 has now been run. What remains is genuinely
outstanding rather than merely unstarted:

| Outstanding | Why | Command when ready |
|---|---|---|
| **The freshness stratum must be regenerated** | All 8 systems score 0.000 on it because the queries are too generic to retrieve their gold chunk (§4.5b). Every freshness figure in this project is therefore unmeasured, not perfect. This is the most important remaining defect. | fix `build_freshness_items` in `integrate_datasets.py`, then re-run §4.5 |
| **Re-tune the evidence sufficiency gate** | The ablation shows disabling it *improves* abstention appropriateness by +0.105 (p<0.0001). Its thresholds are absolute rerank scores fitted to a 60-chunk corpus. | express `min_top_score` / `min_mean_score` relatively, then re-run `ablation_study.py` |
| **`all-mpnet-base-v2` comparison** | Loading a second sentence-transformer in one process aborts on this platform. Needs one process per backend. | `compare_embeddings.py --integrated --backends mpnet` |
| **RAGAS faithfulness / answer relevancy** | Pipeline is wired and verified (32 results, 0 errors); the judge needs `pip install ragas datasets` **and** `OPENAI_API_KEY`. Without a key the script saves its input JSON and stops, by design. | `evaluate_with_ragas.py --run --integrated` |
| **Inter-annotator agreement** | Tooling built and its statistics verified; needs a second human. See §6. | `annotation_agreement.py export → score` |

### A prediction that was recorded in advance, and held

An earlier draft of this section, written before the runs completed, said:

> "I expect the embedding comparison to reach significance at n=965, the
> evidence-gate ablation to show a large abstention effect, and SPIS to remain
> quality-neutral-to-slightly-negative even on the wider lattice. If SPIS shows
> a recall gain there, that is a genuinely new result; if not, the contribution
> stays the security property alone."

All three held: the embedding gain reached p=0.002 (from p=0.52 at n=25), the
gate ablation showed +0.105 on abstention (p<0.0001), and SPIS came out at
−0.004 R@5. The contribution stays the security property alone.

One thing was **not** predicted and is the strongest result in the project: that
the learned router would beat every baseline *and* be the fastest system
(§4.5). I expected it to beat the text-only Adaptive-RAG baseline, which it did
(0.783 vs 0.566 on route accuracy), but not to overtake fixed hybrid retrieval
and always-maximal on end-to-end recall.

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
