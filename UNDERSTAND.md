# Understanding AHRAG

A complete explanation of this project: the problem it addresses, the idea
behind it, how the system works, what was built, what was measured, and what
is honestly still open.

Written to be read start to finish by someone who has not seen the code. No
prior knowledge of the codebase is assumed; RAG and access-control basics are
explained where they matter.

**Other documents, once you want detail:** `FINDINGS.md` (every measured
result), `RUNNING.md` (how to run and test it), `RESEARCH_LIMITATIONS.md` (what
this does *not* establish), `INVENTION_DISCLOSURE.md` (which mechanisms might
be differentiating), `README.md` (design walkthrough).

---

## Part 1 — The problem

### What ordinary RAG does

Retrieval-Augmented Generation is a simple, powerful idea. Instead of asking a
language model to answer from memory — where it may confidently invent things —
you first *retrieve* relevant documents, then ask the model to answer using
only those documents. The retrieved text grounds the answer, and citations make
it checkable.

The standard pipeline is three steps:

```
question  ->  retrieve top-k passages  ->  generate an answer from them
```

This works well for a public knowledge base. It breaks down inside a company,
for reasons that have nothing to do with model quality.

### Why an enterprise is different

Four things are true of a real corporate document store that are not true of
Wikipedia:

**1. Not everyone may read everything.** A finance forecast is restricted to
the finance team. An engineering runbook is restricted to engineers. If the
retrieval step ignores this, the system becomes a tool for reading documents
you are not cleared for — and it will do so *helpfully*, with citations.

**2. Documents supersede each other.** The leave policy from 2023 still exists
in the store next to the 2025 revision. Both are relevant to "how much leave do
I get?". One of them is *wrong now*. A retriever that ranks on similarity alone
has no reason to prefer the current one, and may well prefer the older one if it
happens to be worded closer to the question.

**3. Sources disagree.** Two authorised documents can state different numbers
for the same thing. Silently picking one and presenting it as fact is worse than
saying "these two sources disagree, and here is which one is current."

**4. Sometimes the honest answer is "I can't tell you."** Either the evidence
is not there, or it is there but you are not cleared to see it. In both cases
the correct behaviour is refusal — and crucially, the refusal must not *leak*
which of the two it was. "I can't find anything on that" and "there is a
restricted document about that" are very different disclosures.

### The specific failure this project is built around

Modern RAG systems have started to *route* queries: send simple questions down
a cheap retrieval path, complex ones down an expensive one. The reference work
here is Adaptive-RAG (Jeong et al., NAACL 2024), which trains a classifier to
predict query complexity and picks a strategy accordingly.

Routing on complexity alone misses something in an enterprise, and the miss is
not a detail:

- A *linguistically simple* question can be unanswerable, because the person
  asking cannot read the one document that answers it.
- A *linguistically complex* question can warrant immediate refusal, for the
  same reason.
- And the sharp one: if authorisation is expressed as a **penalty** in a
  cost/quality trade-off, then a large enough efficiency saving can, in
  principle, outweigh it. That is a governance failure written into the
  architecture rather than introduced by a bug.

---

## Part 2 — The idea

> **Route selection should be constrained by what the asker is authorised to
> read, and optimised for evidence sufficiency, freshness, source authority,
> latency and cost — not by query complexity alone.**

Three commitments follow, and they are what the system is built to demonstrate.

### Commitment 1 — Authorisation is a hard filter, not a score

Before anything else happens, the asking user's roles are resolved into an
**authorised scope**: the exact set of text chunks they may read. That set is
then passed as a mandatory allow-list to every retriever. A retriever called
without one searches nothing.

The consequence is worth stating precisely: an unauthorised chunk cannot appear
in an answer, a citation, or a log — **because it was never a candidate**. It is
not filtered out at the end; it never enters. Cost-optimal routing cannot trade
governance for speed, because that trade is not expressible in the formulation.

### Commitment 2 — Route adaptively, among five options

| Route | What it does | When it is right |
|---|---|---|
| **R0** | Abstain, or ask a clarifying question. Never answers from model memory. | No authorised evidence, or evidence too weak |
| **R1** | Sparse keyword search (BM25) | Exact identifiers, error codes, clause numbers |
| **R2** | Dense vector search | Conceptual and paraphrased questions |
| **R3** | Both, fused (Reciprocal Rank Fusion) | Mixed questions — the safe default |
| **R4** | Decompose the question, retrieve iteratively | Comparisons, multi-hop, temporal |

R4 is powerful and expensive. R1 is cheap and brittle. The point of a router is
to spend effort only where it buys something.

### Commitment 3 — Refuse and disclose rather than guess

If the evidence is insufficient or self-contradictory, the system produces a
**typed refusal** — a machine-readable reason, not a vague apology — or a
clarifying question. Where two authorised sources disagree, the disagreement is
*surfaced*, with a stated preference for the current version and a warning
attached to the superseded one.

---

## Part 3 — How the system works

One query, end to end. This is `AHRAGEngine.answer()` in
[ahrag/pipeline.py](ahrag/pipeline.py).

```
 1. Resolve the user  ->  AuthorisedScope          (ACL first, before anything)
 2. Normalise the query against conversation history
 3. Cheap "probe" retrieval, restricted to that scope
 4. Extract 23 router features from query + scope + probe
 5. ROUTE                                          (see Part 4)
 6. Retrieve under the chosen route, inside the scope only
 7. Pack evidence, applying freshness and authority
 8. Detect conflicts; compose freshness warnings
 9. Check sufficiency  ->  abstain or clarify if it fails
10. Generate from the evidence pack only; verify every citation
11. Write an audit record — without storing the source text
```

Two steps carry the design.

**Step 1 is first for a reason.** Authorisation bounds the candidate set
*before* route selection, so route choice is constrained by scope rather than
merely informed by it.

**Step 3 — the probe — is what makes the router governance-aware.** It is one
shallow keyword lookup plus one shallow vector lookup, both already restricted
to the user's scope. It answers a question no query-text classifier can:
*does the corpus this person is allowed to read contain anything relevant at
all?* That is what separates "hard question" from "question this user has no
evidence for". Measurement later showed this is the single most important
mechanism in the whole system.

### A worked example

Ask *"What is the Q3 revenue forecast?"* as two different people:

| | Dan (finance analyst) | Erin (external contractor) |
|---|---|---|
| Authorised scope | includes the restricted forecast | excludes it |
| Probe finds | strong match | nothing relevant |
| Route chosen | **R3** — hybrid retrieval | **R0** — abstain |
| Result | answers, citing the forecast | *"I can't answer that from the sources you're authorised to read."* |

Erin's response contains **zero** evidence rows and no mention that a
restricted document exists. The route, the evidence and the answer are all
functions of *who is asking*. That is the thesis in one example.

---

## Part 4 — The architectural centrepiece

Routing is a **two-layer** decision, and the layering is the contribution.

```
        ACL pre-filter  ->  AuthorisedScope
                 |            (hard, upstream, NOT a term in any formula)
                 v
        admissible route set
                 |            governance predicates:
                 |              - is the scope empty?
                 |              - is probe confidence below the floor?
                 v
        ranking, over the admissible routes ONLY:

           P1  hand-tuned utility:  U = Q - λ_L·L - λ_C·C - λ_R·R
           P2  trained classifier:  23 features, incl. governance signals
           B6  trained classifier:  18 text-only features  (= Adaptive-RAG)
```

Read the diagram bottom-up and the claim becomes clear. Admissibility is
computed **before** any preference is evaluated. So no value of the quality,
latency or cost terms can reinstate a route that governance excluded. It is a
structural property, not a tuning outcome.

This also answers the obvious objection to adding machine learning. A trained
model that has learned to love the expensive route still cannot answer for a
user with an empty scope, because **the model ranks the admissible set; it does
not decide admissibility.** It can change which permitted route wins. It cannot
widen what is permitted. That is asserted by tests
(`tests/test_routing.py::TestLearnedRouter`), not merely claimed in prose.

### How this compares to the alternatives

| System | Routes on | Governance-aware? |
|---|---|---|
| Plain RAG | nothing — one fixed pipeline | no |
| Adaptive-RAG (Jeong et al. 2024) | trained query-complexity classifier | no |
| **AHRAG P1** | hand-tuned utility over governance + probe + cost | yes |
| **AHRAG P2** | **trained classifier** over the same signals | yes |

The comparison that isolates the contribution is **P2 versus B6**: same corpus,
same queries, same training labels, same model family. The *only* difference is
whether the classifier is allowed to see the governance and probe features.

---

## Part 5 — What was built in this phase

The project arrived with a working prototype and a review document,
`improvement.txt`, which argued that none of its research claims were supported
by evidence — the corpus was 9 documents, the evaluation was 32 queries, and the
router's weights were hand-guessed and never fitted to anything. The task was to
fix that.

### First: an audit, which changed the plan

The previous phase's scripts *ran*. What they produced was never used:

- All three consumers of the evaluation set called it with no arguments, so
  every reported number came from the 9-document demo corpus while the logs
  described a 1,125-document one.
- **83% of the generated answer keys pointed at the wrong text.** The converter
  invented chunk IDs by page number; the system assigns them by 900-character
  section split. A wrong ID reads out as "score 0", not as an error, so the
  numbers looked plausibly bad rather than broken.
- The trained router existed only as a file on disk — no code path reached it.
- The reported 97.2% router accuracy was label leakage: 500 of 532 training rows
  were labelled by a rule over the source dataset's own metadata, extracted
  against an *empty* authorised scope so every governance feature was zero.

### Then: a blocker nobody had hit

`improvement.txt` asked for 5,000–10,000 chunks. The embedding backend could not
build an index at that size at all: it constructed a dense term-document matrix
(42,719 × 7,082 — **1.2 GB**) and ran a full singular value decomposition on it,
which does not finish.

Replaced with a **randomized truncated SVD** that never materialises the matrix.

| | Before | After |
|---|---|---|
| Index build, 7,082 chunks | did not complete (>600 s) | **14.1 s** |
| Query latency | — | **30 ms** |
| Agreement with the exact method | — | **identical** (top-5 overlap 5/5) |

Nothing else in the work programme was runnable until this was fixed.

Three crashes were also diagnosed: a macOS process abort from two libraries each
bundling their own OpenMP runtime and being loaded in the wrong order; a test
suite segfault that appeared the moment an optional neural dependency was
installed; and a training crash from parallel workers on Python 3.14.

### Then: the actual work

**A real corpus.** 9 documents → **6,139** (7,082 chunks), built from
FinanceBench (SEC filings), PolicyQA (privacy policies) and HotpotQA
(multi-hop reference text).

> **Stated plainly:** 5,890 of those are single-paragraph reference articles,
> included so that first-stage retrieval is genuinely *selective*. Only **249**
> are enterprise-shaped (165 policy, 84 finance). 249 is the number to compare
> against the review's "500–1000 documents", and it does not meet it. What the
> corpus does deliver is retrieval difficulty and ACL structure.

**Answer keys that are actually correct.** Every gold chunk is now located by
*running the real chunker* and finding the evidence span inside the resulting
chunks. Items whose evidence cannot be located are **dropped, never guessed** —
550 of 1,200 candidates were dropped on that basis, and the count is reported.

**Governance structure the metrics can exercise.** 9 audience classes across
the corpus giving **7 distinct principal scopes**, **15 supersession chains**,
**60** permission-boundary probes (the answer exists, the asker may not read it),
**30** genuinely unanswerable questions, **40** adversarial variants
(misspellings, rambling phrasing).

**A real learned router.** All five routes are now *actually executed* on all
965 queries — the previous code looped over routes but called a function that
ignored the route argument. Each query is labelled with the cheapest route that
achieved the best measured recall. Split 60/20/20, test partition touched once.

**A proper Adaptive-RAG baseline (B6).** The old B5 was two hand-set token-count
thresholds described as "the Adaptive-RAG premise". B6 is the real shape of it:
a *trained* classifier over query-text features only.

**Statistics that are what they claim.** The previous paired significance test
filtered missing values per system independently and then truncated, which pairs
query *i* against query *j* — silently no longer paired. Its p-value counted
resamples where the difference was ≤ 0, which is a posterior quantity, not a
p-value. `ahrag/stats.py` now provides one correct implementation and *refuses*
misaligned input.

---

## Part 6 — What the measurements show

Full detail in `FINDINGS.md`. Charts in `results/`.

### The learned router wins; the hand-tuned one does not

965 queries, aligned on the 875 where every system produced a value. Every
system shares one corpus, one index, one generator, one ACL layer — **only the
routing policy differs.**

| System | Recall@5 | Latency | ACL violations |
|---|---|---|---|
| **P2 learned router** | **0.548** | 0.118 s | **0** |
| B3 fixed hybrid | 0.542 | 0.139 s | 0 |
| B6 Adaptive-RAG (trained) | 0.539 | 0.127 s | 0 |
| B5 complexity-only | 0.539 | 0.145 s | 0 |
| B4 always-maximal | 0.537 | 0.142 s | 0 |
| P1 governance (rule-based) | 0.524 | 0.139 s | 0 |
| B2 fixed dense | 0.519 | **0.113 s** | 0 |
| B1 fixed BM25 | 0.473 | 0.125 s | 0 |

P2 is best on every retrieval metric, and faster than every system except B2 —
which is 0.005 s quicker and 0.029 worse on recall. Against always running the
most expensive route (B4) it is **17% faster with better quality**, achieved by
sending most queries down the cheap keyword path (641 of 965) and abstaining on
127 rather than running 965 iterative retrievals.

> **Treat the latency column as soft.** These runs shared a machine with other
> jobs, and `RESEARCH_LIMITATIONS.md` §2.5 already cautions that latency here is
> an estimate rather than a benchmark. The *route distribution* is the reliable
> evidence for the efficiency claim: P2 spends the expensive route on 2 queries,
> B4 on all 965.

**Zero ACL violations across all eight.** That is the intended result, not a
selling point: enforcement sits upstream of routing, so no routing policy can
break it. The metric exists to *verify* the invariant, and across 965 queries
and 7 principals it holds.

And the uncomfortable half: **P1, the shipped hand-tuned router, is
significantly worse than five of seven comparators.** The architecture is
sound — the same constraints with a fitted ranker produce the best system in the
set — but the hand-picked weights are not. An independent grid search over the
trade-off coefficients ranks the shipped values **7th out of 27**. The review's
"unvalidated priors" charge is confirmed, and the answer is fitting, not more
hand-tuning.

Also worth knowing: B5 (complexity-only) and B6 (real Adaptive-RAG) are
statistically indistinguishable from B3 (a fixed pipeline). **At this scale,
adaptivity by itself buys nothing.** What buys something is adaptivity fitted to
measured outcomes.

### The probe is the load-bearing mechanism

Nine ablations, each disabling exactly one mechanism. One dominates everything:

| Disabled | Recall@5 | Abstention correctness |
|---|---|---|
| *nothing (full system)* | 0.507 | 0.802 |
| **the governance-scoped probe** | **0.013** | **0.104** |

Removing the probe collapses the system: Cohen's *d* = −1.06 on recall and
−1.52 on abstention. The router falls back to refusing 96% of queries, because
without probe evidence the quality model cannot tell that answering is
worthwhile at all. This is the strongest single result in the project.

### A defect the ablations found

Turning the **evidence sufficiency gate off** *improves* abstention correctness
by **+0.105 (p < 0.0001)**. The gate is causing *inappropriate* refusals: its
thresholds are absolute relevance scores tuned on a 60-chunk corpus and they do
not transfer to 7,082 chunks, where scores sit lower. Found by measurement,
recorded, **not yet fixed.**

### Why the evaluation set had to grow

A neural encoder beats the offline default by **+0.0215 Recall@5, p = 0.002**.

The same comparison on the 9-document corpus gave +0.040 at **p = 0.52** —
indistinguishable from noise. At n ≈ 870 the gain is a third the size and
*reliably measurable*. That contrast is the clearest demonstration in the whole
project of why sample size mattered. The effect size is small (*d* ≈ 0.10), so
the honest phrasing is "real but small", not "neural embeddings fix retrieval".

### The learned router versus the real Adaptive-RAG

Held-out route accuracy, test split touched once, n = 129:

| System | Accuracy |
|---|---|
| Oracle (upper bound) | 1.000 |
| **P2 — all 23 features** | **0.783** |
| B6 — Adaptive-RAG, text-only | 0.566 |
| Majority class | 0.651 |

Feature importances corroborate it: the top three are `probe_agreement` (0.215),
`dense_confidence` (0.112) and `intent_comparison` (0.104) — two of the three
are governance/probe signals that B6 is not allowed to see.

And more labels help, then plateau: validation accuracy rises 0.695 → 0.813 as
the training set grows from 38 to 386 queries.

---

## Part 7 — The novel contribution

While tracing the retrieval path, something turned up that was more interesting
than a performance bottleneck.

### The finding

The project's own design document claimed that *"an unauthorised chunk cannot
influence a route choice, a rerank ordering, an evidence pack."* **That was
false as implemented.**

Access control was enforced. **Non-interference was not.** The distinction is
textbook security, and it matters here:

- **Access control** constrains *what comes out*. No restricted text is ever
  returned — that part was true.
- **Non-interference** constrains *what the output depends on*. And the output
  did depend on restricted documents.

Three statistics were computed over the *whole* corpus and only then restricted
at query time:

1. keyword-search term weights (rare terms score higher — but "rare" was
   measured across documents you cannot read),
2. the dense embedding basis, fitted over everything,
3. the reranker's term weights, likewise.

So the *ordering* of authorised results was a function of unauthorised content.
It even reached a governance decision: the confidence value that gates the hard
"is there enough evidence to answer?" threshold is derived from those weights —
meaning a restricted user's **refusal** partly depended on documents they could
not read.

### The fix

Under access control, the corpus is not one corpus. It is a **lattice of
subcorpora**, one per distinct authorised scope. The number of distinct scopes
is small — bounded by role combinations, not headcount. On the benchmark corpus,
7 principals give 7 classes, the narrowest seeing 44% of the corpus.

**Scope-Pure Index Specialisation** fits all three statistics *per class*, over
that class's chunks and nothing else. Every parameter of a class's index is then
a function of that class's chunks alone, so nothing outside it can influence
anything inside it.

### Measured

Hold a principal's authorised subcorpus fixed, delete everything outside it, and
compare. A non-interfering index must return identical results.

| | Top-1 result changed | Evidence order changed |
|---|---|---|
| Default (global statistics) | **6.2%** | **31.2%** |
| Specialised | **0%** | **0%** |

Nearly a third of queries had their evidence ordering changed by documents the
asker cannot read. Specialisation eliminates it.

### And what it costs — stated because it is not free

Non-interference costs **0.0040 Recall@5 (p = 0.027)** — about one item in 250.
The cost does **not** disappear on a wide ACL lattice. So the contribution is
**the security property alone, not a retrieval improvement.** That was written
down as a prediction before the experiment ran, and it held.

There is a tuning parameter, λ, which interpolates back toward the global
statistics. It is deliberately framed as an **information-flow budget** rather
than an ordinary hyperparameter: λ = 0 is provably non-interfering, λ = 1
restores the leak, and the curve between them is a *measured* privacy/utility
frontier an operator can choose a point on. It governs the keyword channel only,
so it is half a frontier, not a complete one.

---

## Part 8 — What went wrong, including our own mistakes

A review should be able to see the errors, not just the results. Three of these
were mine, found and corrected during the work.

**1. My freshness test set was unmeasurable.** All eight systems scored exactly
0.000 on the 15 freshness questions. That was not eight systems failing — my
generator phrased them as *"What is the current rule in {heading}?"*, almost
entirely function words, which cannot retrieve one specific chunk out of 7,082.
It also explains why the freshness metric reported a perfect 1.000 in every
run *including the one that disabled freshness enforcement*: the metric was
never exercised, and an unexercised metric reads as perfect. Fixed — the stratum
went **0/15 → 15/15** — and now guarded by four dataset-integrity tests.

**2. My headline security number was invalid.** I first reported "33.3% of
routes changed" as the interference result. That comparison could never have
been zero for *any* system: deleting unreadable documents also changes
`restricted_fraction`, a signal the router uses *on purpose*. The experiment was
mixing an intended dependence in with the unintended one, and the 0% I got on
the small corpus was luck. Fixed by holding the route fixed, which isolates what
specialisation actually governs — and the claim then replicated properly at
scale.

**3. Two scripts announced work they had not done.** One referenced an
undefined constant that would only fail at the very end of a long run; another
had its file write left behind a conditional, so a plain run produced no output
at all. Two complete benchmark runs finished and wrote nothing before I noticed
the timestamp had not moved. This is precisely the failure mode the audit in
Part 5 criticised, reproduced by me in a new form.

**4. Abstention on unanswerable questions is unreliable.** *"What is the company
policy on interplanetary business travel?"* gets **answered**, because "policy"
and "business travel" match the real travel policy strongly enough to clear both
the probe floor and the sufficiency gate. Four of five obviously-unsupported
questions behave this way. Real defect, documented, not fixed.

**5. Four governance mechanisms are inert** on this corpus — the
restricted-scope signal, the routing freshness penalty, the routing conflict
signal, and the authority gate each produce exactly zero change on every metric.
Two of them demonstrably reach the engine (they shift the route distribution)
but change no outcome.

---

## Part 9 — What this does *not* establish

Being clear about this is what makes the rest credible.

- **P2's lead is not a clean held-out win.** It trained on labels derived from
  40% of the same queries it was then evaluated on, while no other system had
  any training data at all. The route-accuracy comparison in Part 6 *is*
  properly held out; the end-to-end recall comparison is not.
- **No inter-annotator agreement figure exists.** The answer keys are
  programmatic. The tooling and the statistics are built
  (`annotation_agreement.py`, with Cohen's κ, Krippendorff's α and bootstrap
  intervals), but it needs a **second human** annotator and none has run. A
  model re-labelling its own labels would measure self-consistency and reporting
  it as agreement would be worse than reporting nothing.
- **No human quality judgement.** No Likert-scale rating of answer quality by
  real judges.
- **Freshness does not discriminate.** All eight systems score 1.000, so the
  supported claim is "freshness handling works", not "ours is better".
- **Absolute recall of ~0.52 is dominated by one stratum.** Privacy-policy
  comprehension is 300 of 965 queries at ~0.16 recall. That is genuine
  difficulty, not a defect, but it is why the headline number is not higher.
- **The corpus is not 6,139 *enterprise* documents.** It is 249, plus 5,890
  reference paragraphs.
- **Nothing here has been assessed for novelty or patentability.** See
  `INVENTION_DISCLOSURE.md`, which rates most of its own mechanisms as likely
  unpatentable and says so.

---

## Part 10 — Seeing it for yourself

```bash
python -m pytest tests/ -q                         # 294 tests
python -m pytest tests/test_demonstration.py -s    # a printed guided tour
```

`tests/test_demonstration.py` is the fastest way to understand the system: ten
scenarios that each *print their evidence*, so the run reads as a report rather
than a pass/fail list. It covers the ACL boundary, governance outranking
utility, conflict disclosure, freshness labelling, abstention (including where
it fails), citation verification, route variation, audit minimisation, the ACL
lattice, and non-interference.

For the web app — two pages of setup, six hand-checks with the output to expect,
and a troubleshooting table — see **`RUNNING.md`**.

### The file map

| Area | Where |
|---|---|
| The engine, end to end | `ahrag/pipeline.py` |
| The ACL pre-filter | `ahrag/governance/acl.py` |
| The routers (P1, P2, B5, B6, fixed) | `ahrag/routing/router.py` |
| The 23 router features | `ahrag/routing/features.py` |
| Per-route retrieval | `ahrag/retrieval/pipeline.py` |
| **The ACL lattice** | `ahrag/index/lattice.py` |
| **Scope-pure specialisation** | `ahrag/index/scoped.py` |
| Statistics | `ahrag/stats.py` |
| The eight compared systems | `ahrag/eval/systems.py` |
| Corpus + label plumbing | `ahrag/eval/harness.py` |
| Corpus construction | `improvement_files/datasets/integrate_datasets.py` |
| Router training | `improvement_files/ml_router_training/train_router.py` |
| System comparison | `improvement_files/baseline_code/compare_baselines.py` |
| Ablations | `improvement_files/baseline_code/ablation_study.py` |
| Embedding comparison | `improvement_files/evaluation_tools/compare_embeddings.py` |
| Specialisation measurement | `improvement_files/evaluation_tools/measure_specialisation.py` |
| Charts for slides | `results/` |

---

## In one paragraph

AHRAG is a retrieval-augmented generation system for corporate document stores,
built around the position that *who is asking* should constrain *how the system
searches* — not merely filter what it returns. Authorisation is resolved into a
per-person scope before routing happens, so an unauthorised document is never a
candidate for anything; within that boundary the system chooses among five
retrieval strategies, prefers current documents over superseded ones, discloses
disagreement between sources rather than resolving it silently, and refuses
rather than guessing when the evidence is not there. This phase replaced a
9-document demo with a 6,139-document benchmark and 965 checked answer keys,
fixed a scalability defect that made that corpus impossible to index at all,
trained the router on outcomes actually measured by running every route on every
query, and built a proper Adaptive-RAG baseline to compare against. The learned
router is the best of eight systems and the fastest; the hand-tuned one it
replaces is significantly worse than most baselines, which was the review's
central charge and is now confirmed. Along the way the retrieval statistics
turned out to leak: the *ordering* of results a person was allowed to see
depended on documents they were not, and closing that — at a measured cost of
one item in 250 — is the project's novel contribution.
