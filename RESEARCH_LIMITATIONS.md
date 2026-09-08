# Research Limitations

What this prototype does **not** establish, and what would be required to
establish it.

This document exists because the easiest way to mislead with a research
prototype is to let a working demo stand in for evidence. AHRAG runs, its tests
pass, and its evaluation produces real numbers — none of which means the
underlying hypotheses have been validated.

> **Read this alongside `FINDINGS.md`.** Sections 2 and 3 below were written
> against the 9-document demo corpus, and several of their conclusions have
> since been superseded by a 6,139-document / 965-query benchmark. Where a
> status has changed, it is marked **UPDATED** and points at the new evidence.
> Two conclusions moved in the project's favour (§2.3, §4.1) and one moved
> against it (§5.3, the freshness metric). The demo-corpus caveats themselves
> all still stand for demo-corpus numbers.

---

## 1. The separation this project maintains

| Category | Status | Where the evidence is |
|---|---|---|
| **Implemented functionality** | Built, tested, reproducible | `README.md`, 247 passing tests |
| **Research hypotheses** | **Not validated** | This document, §3 |
| **Potentially differentiating mechanisms** | **Not assessed for novelty** | `INVENTION_DISCLOSURE.md` |

Nothing in this repository should be read as a claim that AHRAG is novel,
patentable, state of the art, or legally protectable.

---

## 2. Limitations of the evaluation

### 2.1 The corpus is a demo, not a benchmark

9 documents, 60 chunks, roughly 15,000 words, written specifically for this
prototype. It contains a planted version conflict, a planted restricted
document, and planted error codes because those are the behaviours being
demonstrated.

**Consequence:** every number in the evaluation is a property of *this corpus*.
None of them transfer. In particular, a corpus with planted conflicts will
naturally show good conflict detection.

**To fix:** run against EnterpriseRAG-Bench (≈500k synthetic company documents
across nine source types) and FinanceBench (10,231 financial QA items with
evidence strings), plus a de-identified corpus from a real organisation with
genuine ACLs. The research draft's §6 dataset plan describes this properly.

### 2.2 The labelled suite is small and single-annotator

32 items, labelled by one author who had already read the corpus. There is no
second annotator, no adjudication, and therefore no inter-annotator agreement
figure.

**Consequence:** the labels carry unmeasured annotator bias, and the sample is
far too small for statistical significance. A 0.03 difference in Recall@5
between two systems on 25 answerable items is well inside noise.

**To fix:** 300–500 items, two independent annotators plus adjudication, reported
inter-annotator agreement, and paired bootstrap confidence intervals on every
comparison. **No significance test is reported here because none would be
meaningful at n=25.**

### 2.3 The systems are statistically indistinguishable on retrieval

The measured result is that P1 **ties** B4 and B5 on Recall@5, MRR, and nDCG. It
does not beat them.

The reason is structural: on a 9-document corpus, almost every answer is
findable by almost every route. The suite cannot separate retrieval policies
because there is not enough retrieval difficulty to separate them with. The
differences that *are* visible — abstention appropriateness, route distribution,
estimated cost — are the ones the corpus can actually exercise.

**A tie is the honest reading. It is not evidence that the router works, and it
is not evidence that it does not.**

**UPDATED at benchmark scale (`FINDINGS.md` §4.5).** With 6,139 documents and
965 queries the systems separate cleanly, and the separation is not flattering
to the shipped router. The learned router P2 leads on every retrieval metric
(R@5 0.530) while being the fastest; the hand-tuned P1 (0.507) is
*significantly worse* than B3, B4, B5, B6 and P2. So the tie was indeed a
corpus artefact, and once removed the finding is that the **architecture**
holds up while the **hand-picked utility weights** do not.

### 2.4 Retrieval depth was chosen to make the comparison non-vacuous

With production-scale `candidate_top_k` (30–100), the first-stage retriever
returns nearly the whole authorised pool, the reranker then sees identical
candidates whatever route ran, and **all six systems produce byte-identical
results**. This was observed during development, not hypothesised.

`candidate_top_k` is therefore set to 10, preserving roughly the same
first-stage *selectivity ratio* (~20% of the authorised pool) that a production
system has against a large corpus.

**Consequence:** this is a defensible choice, but it is a choice made to keep the
experiment meaningful, and results at this depth may not hold at other depths.

### 2.5 Latency and cost figures are near-meaningless at this scale

Every system runs in ~1 ms. The p95 spread across systems is smaller than
measurement jitter, so the latency column cannot support any claim.

Cost is an **estimate** computed from per-route token budgets and local price
constants. The default extractive generator makes no API calls and incurs no
actual cost. The estimate is labelled as such everywhere it appears, and it is
useful only for comparing routes against each other under a fixed accounting
assumption.

**To fix:** measure on a corpus large enough for retrieval to dominate, with a
real LLM in the loop and real token accounting.

### 2.6 Route agreement is a diagnostic, not a score

`expected_route` in the eval set is a *hypothesis* about which route should win.
It is used only for the route-agreement diagnostic (P1 scores 0.53) and feeds no
retrieval or answer metric. No system is rewarded for matching it. Treating
route agreement as an accuracy measure would be circular: it would score the
router against the author's intuition rather than against outcomes.

---

## 3. Hypotheses that remain open

These restate the research draft's H1–H3 with an honest status.

### H1 — Context-aware routing improves evidence recall and faithfulness over fixed sparse-only and dense-only RAG

**Status: not supported, not refuted.** P1 (0.840) ties B4/B5 and edges B1/B3
(0.820) on Recall@5, but the margin is one item on a 25-item denominator.
Faithfulness is not measured at all: the extractive generator is faithful by
construction, so a faithfulness metric here would measure the generator's
architecture, not the retrieval policy.

**What would test it:** the larger annotated corpus from §2.2, with a real
generator, RAGAS/ARES faithfulness scoring, and claim-level entailment via
RAGChecker.

### H2 — Adaptive hybrid retrieval achieves comparable or better quality than an always-maximal pipeline at lower mean latency and token cost

**Status: weakly consistent, at a scale that cannot confirm it.** P1 matches B4's
Recall@5 (0.840) at a lower estimated cost per query (0.003687 vs 0.003750).
The direction is right. The magnitude is under 2%, the latency component is
noise (§2.5), and the cost is an estimate (§2.5).

**What would test it:** a corpus where R4's decomposition genuinely costs
multiple seconds and multiple thousand tokens, so the saving is measurable
rather than inferred.

### H3 — Corpus metadata (authority, timestamp, type, permissions) improves citation quality and reduces unsupported or unauthorised answers

**Status: partially supported for the permissions component; untested for the
rest.**

Supported: P1 has the best abstention appropriateness (0.875 vs 0.812–0.844),
and the per-type breakdown attributes the entire gap to `permission_constrained`
items. P1 is the only system that selects R0, because it is the only one that
can see that the authorised corpus is thin.

Untested: the *authority* and *timestamp* components. Freshness compliance is
1.000 for all six systems, which means the metric never discriminated — the
corpus has exactly one supersession chain, and every system happened to lead
with the current version. A metric that is 1.000 everywhere has told you nothing.

Unauthorised answers: ACL violation rate is 0 for all six systems, which is the
required result but is **not evidence for H3**. It holds because ACL enforcement
is architecturally upstream of routing, so routing cannot affect it either way.

**What would test it:** multiple supersession chains, documents with genuinely
varying authority that compete for the same query, and adversarial cases where
the stale document is more lexically relevant than the current one.

---

## 4. Component-level limitations

### 4.1 The router is rule-based, not learned — **NO LONGER TRUE**

The research draft proposes a calibrated gradient-boosted or small-LM
classifier trained on offline route labels. This prototype implements an
interpretable linear model with hand-set weights, because shipping an untrained
model or fabricating training labels would both be worse.

**Consequence:** the weights in `config/router.yaml` encode the author's reading
of the literature's failure modes. They were *not* fitted on the evaluation set —
which protects against circularity but means they are unvalidated priors.

**To fix:** run all candidate routes offline on a training partition, score
evidence sufficiency against annotated references, assign each query the
lowest-cost route that clears pre-registered thresholds, train a calibrated
classifier, and compare it against both this rule-based router and an LLM router
on a held-out test set that was never used to set thresholds.

**DONE — this is exactly what was built** (`FINDINGS.md` §4.1, §4.5). All five
routes are executed on every one of 965 queries via `FixedRouter`; each query is
labelled with the cheapest route achieving the best measured recall; the split
is 60/20/20 with the test partition touched once. `LearnedRouter` (system P2)
ranks the admissible routes with the resulting XGBoost model, and reaches 0.783
held-out route accuracy against 0.566 for a trained text-only Adaptive-RAG
baseline on identical data.

Two things this confirmed about the paragraph above:

1. The "unvalidated priors" concern was correct. An independent λ grid search
   ranks the shipped `0.09/0.35/0.55` **7th of 27**, and P1 is significantly
   worse than five of seven comparators at scale.
2. The rule-based router remains the **default**, because the learned one
   requires the optional `xgboost` extra and degrades to P1 without it. What
   changed is that "adaptive" is now a fitted policy where the extra is
   installed, rather than a heuristic everywhere.

The learned model does **not** get to decide admissibility — it ranks only the
routes the governance constraints already admitted, so it can change which
admissible route wins but cannot widen what is permitted.
`tests/test_routing.py::TestLearnedRouter` asserts that.

### 4.2 The offline embedding backend is LSA, not a neural encoder

The default `lsa` backend is real distributional semantics (TF-IDF + truncated
SVD, queries folded into the learned term space), and it generalises across
paraphrase within the corpus vocabulary. It is not a neural bi-encoder.

**Consequence and evidence:** several measured failures are vocabulary-mismatch
cases that a neural encoder would plausibly bridge — `sem-003` ("unused holiday
… leave the company" vs "accrued but untaken leave … termination") and `sem-005`
("fall ill while on holiday" vs "Sickness … annual leave"). LSA fitted on 60
chunks has too little co-occurrence data to learn those associations.

**That improvement is a hypothesis, not a result.** Installing
`sentence-transformers` is expected to help; nobody has measured whether it does
on this suite.

### 4.3 The relevance threshold cannot separate all cases

`min_top_score` and `min_mean_score` were calibrated by measuring the score
distribution over the labelled suite and cutting between the true-positive and
off-topic clusters. **The clusters overlap**, so no threshold separates them
cleanly. The residual errors were left in the reported results rather than tuned
away:

- `una-003` ("parental leave for adoptive parents") scores 0.307 and is answered
  from general leave-policy text. The corpus has no parental-leave content.
- `acl-003` (contractor asking about `ERR-5041`) scores 0.416 against the Atlas
  status report's *mention* of the code, and is answered instead of refused. No
  restricted content leaks, but a partially-responsive answer is worse than a
  refusal here.
- `sem-005` scores 0.089 and is refused despite being answerable.

This is the central weakness of a purely lexical relevance signal, and it is
exactly the situation a trained cross-encoder exists for.

### 4.4 Conflict detection is pattern-based and narrow

Two detectors: same `policy_family` across different documents (structural), and
same topic + unit + differing value (numeric, via regex over a fixed topic
lexicon).

**It will miss:** contradictions expressed without numbers, contradictions
between documents that were never assigned a shared `policy_family`, semantic
negation ("employees may" vs "employees may not"), and conflicts spanning more
than two versions.

**It could produce false positives** if two unrelated quantities in the same
family shared a topic term — which is why sentences with no recognised topic are
skipped rather than bucketed under a catch-all. A false conflict warning erodes
trust in the true ones.

### 4.5 Query decomposition is rule-based

R4 splits on "between X and Y", explicit comparison connectives, difference
questions, and conjunctions. An LLM decomposer would produce better sub-queries
but would make R4 non-reproducible and put its latency behind an external
service — both fatal to the latency/cost comparison the evaluation exists to
make.

**Consequence:** `cmp-001` and `cmp-003` retrieve only one of the two required
policy editions (R@5 = 0.5). The decomposition is visible in the UI trace, so a
reviewer can see whether a weak answer came from bad decomposition or bad
retrieval — but it is still weak.

### 4.6 Follow-up handling is a heuristic, not coreference resolution

A short pronoun-led query is prefixed with the previous turn. That is all. It
will fail on any follow-up needing real anaphora resolution. The rewritten query
is shown in the trace so the user can see what was actually searched.

### 4.7 ACL model is coarse

Flat role-based access at document granularity. Real enterprise systems have
group hierarchies, attribute-based rules, per-field redaction, time-bounded
grants, delegation, and break-glass access. Chunk-level ACL is inherited
wholesale from the parent document.

**Not modelled at all:** inference attacks. A user could potentially reconstruct
restricted facts from an authorised document that quotes them. The seeded corpus
contains a live instance of this — the restricted forecast and the authorised
Atlas report both discuss the ERR-5041 rate — and AHRAG does nothing about it.

### 4.8 Authentication is simulated

User selection in a dropdown stands in for authentication. There is no identity
provider, no session management, no token validation, and no authorisation
server. **This is a demo affordance and must not be mistaken for a security
control.**

---

## 5. What "zero ACL violations" does and does not mean

The evaluation reports 0 ACL violations across all six systems and all 32 items,
and `tests/test_governance.py` plus `tests/test_evaluation.py` verify it.

**It means:** under the tested queries, on this corpus, with this ACL model, no
unauthorised chunk reached evidence, citations, generation, or the audit log.

**It does not mean:**

- The system is secure. 32 queries is not a penetration test.
- The property holds under adversarial prompting. No prompt-injection testing
  was performed, and the corpus contains no injected instructions.
- The property holds against inference attacks (§4.7).
- The property would survive a more complex ACL model, delegation, or per-field
  redaction.
- Anything about the LLM path. With `ANTHROPIC_API_KEY` set, the evidence pack
  is still ACL-filtered and citations are still verified, but the *generated
  prose* has not been audited for leakage of information the model might infer
  across authorised passages.

The invariant is structural — unauthorised chunks are never candidates — which is
a stronger position than filtering after the fact. It is not a proof.

### 5.1 It also did not mean non-interference, until recently

The claim above concerns *which chunks reach an answer*. It says nothing about
whether unauthorised chunks **influenced** the answer, and by default they do.

BM25 IDF and average document length, the TF-IDF vocabulary and LSA basis, and
the lexical reranker's IDF were all fitted over the **whole** corpus and only
then restricted at query time. The scores of authorised chunks were therefore a
function of unauthorised content. Because `sparse_confidence` derives from the
raw BM25 score and is tested against the hard `min_probe_for_answering`
constraint, this reached a governance decision: a restricted principal's
*abstention* depended on documents it could not read.

Measured by `improvement_files/evaluation_tools/measure_specialisation.py` (E2),
holding a principal's authorised subcorpus fixed and deleting everything outside
it, over 24 observer/query pairs:

| Condition | Top-1 flips | Evidence-order changes | Non-interfering |
|---|---|---|---|
| Unspecialised (default) | 6.2% | **31.2%** | No |
| SPIS, lambda = 0 | **0%** | **0%** | Yes |

Benchmark corpus, 7 ACL classes, 16 observer/query pairs, **route pinned**.
Pinning is required: deleting unreadable documents also changes
`restricted_fraction`, which the router uses deliberately, so an unpinned
comparison cannot isolate index behaviour. An earlier unpinned run on the demo
corpus reported "33.3% route changes / 0% under SPIS"; that comparison was
invalid and the figures should not be used. See `FINDINGS.md` §4.4.

No restricted text is ever returned in either condition, so this is not an
access-control violation. It is an information-flow violation, and the
distinction matters: **access control constrains outputs; non-interference
constrains dependence.** AHRAG enforced the first and, by default, still does not
enforce the second.

`Settings.index_specialisation` fixes it by fitting all three statistics per ACL
equivalence class (`ahrag/index/scoped.py`), asserted by 26 tests in
`tests/test_noninterference.py`. It is **off by default**, so every number
elsewhere in this document and in the README was produced by the interfering
configuration.

### 5.2 Scope-Pure Index Specialisation does not improve retrieval quality here

This is a negative result and is reported as one.

On the 9-document seed corpus, over 32 items (25 answerable), with 2000-sample
paired bootstrap and Cohen's *d*:

| Metric | Unspecialised | SPIS lambda=0 | Delta | p | d |
|---|---|---|---|---|---|
| Recall@5 | 0.840 | 0.840 | +0.0000 | 1.000 | 0.000 |
| MRR | 0.753 | 0.750 | −0.0033 | 0.623 | −0.200 |
| nDCG@10 | 0.758 | 0.756 | −0.0017 | 0.453 | −0.200 |
| Abstention appropriateness | 0.875 | 0.812 | **−0.0625** | 0.247 | −0.254 |

Nothing here is significant, and the point estimates that do move, move the
wrong way. Three things follow:

1. **The purity property is free but not profitable at this scale.** §2.1
   already explains why: at 60 chunks almost everything is findable by any
   route, so changing the *weights* changes the *order* without changing what
   lands in the top 5. The interference result does not depend on corpus size;
   the quality result does.
2. **The abstention regression is real and traced.** It costs 2 of 32 items.
   Because lambda=0.5 and lambda=1 restore MRR and nDCG to baseline exactly
   while abstention stays at 0.812, the cause is the class-local **LSA basis**,
   not the IDF shrinkage. A per-class basis fitted on 35 chunks is weaker than
   one fitted on 60 — the small-sample problem of §4.2, reappearing per class.
3. **The lambda frontier is currently only half a frontier.** `lambda` governs
   the sparse channel only; the dense basis is class-local whenever
   specialisation is on. A complete purity/utility curve needs a dense analogue
   (interpolating the SVD basis, or the IDF inside `LSAEmbedder`), which is not
   implemented.

**To fix:** the same thing §2.1 and §2.2 already require — a corpus of 500+
documents with genuinely varied ACL role-sets, and an evaluation set whose gold
chunk IDs resolve against real chunker output. The 1,125-document integrated
corpus in `improvement_files/datasets/integrated/` cannot serve: every document
grants `employee`, so the lattice collapses to one class and specialisation is
provably a no-op there (E1 reports this rather than letting it pass silently).

### 5.3 The freshness metric is unmeasured, not perfect

Freshness compliance reports **1.000** everywhere — for all eight systems, and
even in the ablation that switched freshness enforcement off. That figure
should be read as *"this metric was never exercised"*.

The benchmark corpus does contain 15 real supersession chains, correctly linked,
and 38/38 of the freshness items' gold chunks were verified to be the current
version. The defect is in the **queries**:
`build_freshness_items` in `integrate_datasets.py` phrases them as *"What is the
current rule in {heading}?"*. Against 7,082 chunks a query that generic cannot
retrieve one specific chunk regardless of routing policy, and the stratified
results confirm it — **all eight systems score exactly 0.000 Recall@5 on all 15
freshness items**.

**Consequence:** no claim about freshness handling in this project is supported
by evidence. That includes the claim in `INVENTION_DISCLOSURE.md` §8(e) form
("freshness-aware evidence packing correctly prefers current versions in P% of
cases"), which remains untested at scale. The demo-corpus behaviour *is*
verified — `tests/test_scenarios.py` and `tests/test_demonstration.py` both
confirm the current version is preferred and the superseded one is labelled —
but that is one supersession chain, which is the §2.1 problem again.

**To fix:** generate freshness queries from distinctive content *inside* the
versioned section rather than from its heading, so the query can actually
retrieve its own gold chunk, then re-run the comparison.

---

## 6. Engineering limitations

- **Scale.** Everything is in-process and in-memory. The numpy vector store does
  an exact scan; BM25 is rebuilt wholesale on every ingest. Fine at 60 chunks,
  useless at 10⁶.
- **Concurrency.** SQLite in WAL mode with a process-level lock. Not designed for
  multi-worker deployment.
- **Index freshness.** Any ingest triggers a full index rebuild. No incremental
  update.
- **No streaming.** Answers are returned whole.
- **No feedback loop.** The research draft's §5.1 stage 6 proposes using reviewed
  failures to update router thresholds. Not implemented — thresholds are static
  config.
- **Uploads are trusted.** File type and size are checked; content is not
  scanned. Do not point this at untrusted input.

---

## 7. Honest summary

This prototype demonstrates that a governance-aware routing architecture **can be
built** and that its governance invariant **holds under test**. It shows one
measurable behavioural difference attributable to the governance signal: P1 is
the only system that selects abstention as a route when the authorised corpus is
thin, and it scores best on abstention appropriateness as a result.

It does **not** demonstrate that governance-aware routing improves retrieval
quality, that it meaningfully reduces latency or cost, or that any of it
generalises beyond a 9-document demo corpus. Those remain hypotheses, and the
work required to test them is described above.
