# Invention Disclosure — Preliminary Technical Record

**Subject:** AHRAG — governance-constrained adaptive retrieval routing
**Status:** Engineering record prepared for review. **Not a legal opinion.**
**Prepared:** 2026-08

---

## ⚠️ Read this first

This document does **not** claim that anything described here is novel,
inventive, non-obvious, patentable, or legally protectable. It has not been
prepared or reviewed by a patent professional, and no prior-art search has been
conducted beyond reading the papers listed in §4.

Its purpose is narrow: to record, in enough technical detail to be assessed,
(a) which mechanisms **might** be differentiating, and (b) the prior art that a
qualified attorney would need to weigh against each one. Several mechanisms
below are assessed by the author as **likely unpatentable**, and that assessment
is stated plainly rather than omitted.

**Required next steps before any protection is pursued:**

1. Engage a registered patent attorney or agent.
2. Commission a professional prior-art and freedom-to-operate search covering
   patents and published applications (not just academic literature) in
   information retrieval, access control, and LLM orchestration.
3. Establish whether any subject matter survives §101 / Art. 52 EPC eligibility
   analysis for software-implemented methods in the relevant jurisdictions.
4. Confirm publication and disclosure status — public disclosure may already
   have started grace-period or absolute-novelty clocks.
5. Review employer/institution IP assignment obligations.

---

## 1. Field

Retrieval-augmented generation over access-controlled, versioned enterprise
document corpora, where a query router selects among retrieval strategies under
operational and governance constraints.

---

## 2. Problem addressed

Existing adaptive-RAG systems select a retrieval strategy from properties of the
**query text** — typically predicted complexity. In an enterprise setting the
appropriate strategy also depends on properties of the **requesting principal**
and the **authorised subset of the corpus**:

- A linguistically simple query may be unanswerable because the asking user
  cannot read the canonical source.
- A linguistically complex query may warrant immediate abstention for the same
  reason.
- Cost-optimal routing can, in a naïve design, select a cheap path that draws on
  material the user is not permitted to see.

The last point is the sharp one. If authorisation is a *penalty term* in a
utility function, then a sufficiently large latency or cost saving can, in
principle, outweigh it. That is a governance failure expressible in the
architecture itself.

---

## 3. Mechanisms that may be differentiating

Each is described precisely enough for assessment, with the author's own
prior-art risk rating.

### M1 — Authorisation as a hard admissibility constraint upstream of route selection

**Description.** Before any routing computation:

1. The requesting principal's roles are resolved to a set of permitted chunk
   identifiers (`AuthorisedScope`).
2. That identifier set is passed as a mandatory allow-list argument to every
   retriever. A retriever invoked without one searches nothing.
3. The router computes an admissible route set from governance predicates
   (empty scope; probe confidence below threshold) **before** any utility is
   evaluated.
4. Utility maximisation runs only over the admissible set.

**The claimed property:** because step 3 precedes step 4, no value of the
latency, cost, or quality terms can reinstate a route excluded on governance
grounds. Cost-optimal routing cannot trade governance for efficiency, because
that trade is not representable in the formulation.

**Implementation:** `ahrag/governance/acl.py`, `ahrag/routing/router.py`
(`_admissible_routes` runs before `_score_route`).
**Verified by:** `tests/test_routing.py::test_high_confidence_cannot_outbid_empty_scope`,
which constructs a case where the excluded routes carry higher raw utility and
still lose.

**Prior-art risk: HIGH.** Pre-filtering search by ACL is standard practice in
enterprise search (SharePoint, Elasticsearch document-level security, Glean, and
essentially every commercial enterprise-RAG product). HONEYBEE and similar work
address RBAC partitioning in vector databases directly. The *ordering* relative
to an adaptive router may be a narrower point, but "apply access control before
search" is neither new nor non-obvious on its own. **The author's assessment is
that M1 alone is very unlikely to be patentable.** Its value here is as a
correctness property, not as IP.

---

### M2 — Governance signals as inputs to the route-utility computation

**Description.** Beyond the hard constraint, three governance-derived quantities
enter the *risk* term `R(z|x)` of the utility function:

- `restricted_fraction` — the share of the corpus withheld from this principal.
  A heavily filtered view may be a partial picture of the topic, so evidence risk
  rises for every answering route while abstention's risk is held at its base
  rate.
- `low_probe` — evidence risk grows linearly as the best **ACL-scoped** probe
  result falls below a reference confidence.
- `conflict_likelihood` — a detected multi-version condition raises risk
  specifically for routes that cannot surface both sides (single-family R1/R2).

**The claimed property:** the same query from two principals with different
authorisations can select different routes — including selecting abstention for
one and retrieval for the other — because route utility is a function of the
principal's scope, not only of the query text.

**Implementation:** `ahrag/routing/router.py::_estimated_risk`,
`ahrag/routing/features.py`.
**Observed in evaluation:** P1 is the only one of six systems that ever selects
R0, and its entire abstention-appropriateness advantage comes from
permission-constrained items.

**Prior-art risk: MEDIUM.** Adaptive-RAG, Self-RAG, CRAG, HyPA-RAG, and R³AG all
route or gate retrieval; none of the reviewed papers conditions the route
decision on the requesting principal's authorised corpus share. Cost-aware
routing work optimises quality against latency/cost without governance terms.
The specific composition — *governance-derived risk terms inside a cost-aware
routing utility* — is where any differentiation would sit. Whether that
composition is non-obvious over the combination is exactly the question an
attorney must answer, and combinations of known techniques frequently fail the
obviousness test.

---

### M3 — Governance-scoped probe as a routing feature

**Description.** Before routing, a deliberately cheap top-*k* retrieval is run in
**both** families, restricted to the authorised scope. Its outputs — normalised
sparse confidence (blending saturated top score with margin over the runner-up),
dense confidence, family agreement, and multi-version detection — become router
features.

**The claimed property:** the router observes whether the *authorised* corpus
actually contains relevant material before committing to a route. This lets it
distinguish "this question is hard" from "this principal has no evidence for
this question" — a distinction a text-only router cannot make, and the one that
makes governance-grounded abstention possible.

**Implementation:** `ahrag/retrieval/pipeline.py::probe`,
`ahrag/routing/features.py::ProbeSignals`.

**Prior-art risk: MEDIUM-HIGH.** Two-stage retrieval and retrieval-confidence
gating are well established; CRAG's retrieval evaluator is directly comparable,
and query-performance prediction is a mature IR subfield. The narrower point is
that the probe is *scope-restricted*, so its confidence is a statement about the
principal's authorised view rather than the corpus. That is a small delta over
"run a cheap retrieval first, then decide".

---

### M4 — Structural conflict disclosure with non-destructive preference ordering

**Description.** When authorised evidence contains multiple versions of one
topic, the system:

1. Detects it structurally (shared `policy_family`, different parent documents)
   **and** semantically (same topic + unit, differing numeric value).
2. Computes a *preferred* source by a stated ordering: not-superseded, then
   latest already-effective date, then authority score.
3. **Retains the non-preferred source** in the evidence pack and names it, with
   both chunk IDs, in the rendered answer.

**The claimed property:** conflict is disclosed rather than resolved. From the
user's side, silent resolution is indistinguishable from the system never having
found the other version — so the preference ordering is surfaced as a
recommendation, not applied as a filter.

**Implementation:** `ahrag/evidence/conflict.py`,
`ahrag/governance/freshness.py::preferred_in_family`.
**Verified by:** `tests/test_evidence.py::test_both_sides_stay_in_the_report`.

**Prior-art risk: MEDIUM.** Contradiction detection in NLI and multi-document
summarisation is extensive; document versioning and effective-dating are
standard in content management. The combination — metadata-driven version-family
detection, numeric-claim contradiction extraction, and a mandatory
non-destructive disclosure policy in the generated answer — may be narrower, but
each component is individually well known.

---

### M5 — Abstention-reason typing coupled to appropriateness measurement

**Description.** Every refusal carries a typed reason from a closed enumeration
(no authorised evidence / insufficient evidence / low relevance / insufficient
source diversity / low authority / needs clarification / route is
non-generative). The evaluation harness scores abstention appropriateness
**symmetrically** — refusing an answerable question is penalised exactly as much
as answering an unanswerable one — so a trivially over-refusing system cannot
score well.

**The claimed property:** abstention becomes a measurable, attributable decision
rather than an undifferentiated failure mode, and the reason type localises the
failure (retrieval vs governance vs ambiguity).

**Implementation:** `ahrag/models.py::AbstentionReason`,
`ahrag/evidence/sufficiency.py`, `ahrag/eval/metrics.py::abstention_appropriate`.

**Prior-art risk: HIGH.** Typed refusal reasons and selective-prediction metrics
are standard in QA and calibration literature. **Likely a methodological
contribution rather than an invention.**

---

### M6 — Route-decision auditability under content minimisation

**Description.** Each execution writes an audit record carrying the full route
decision — every feature value, the complete candidate-utility table with
per-term penalty decomposition, applied hard constraints, retrieval timings,
document identifiers considered and used, ACL-withheld count, citations,
abstention reason, cost estimate — while persisting only a SHA-256 hash of the
query and **no** raw query or evidence text. Raw-text capture is an explicit
opt-in flagged on every affected record.

**The claimed property:** a route decision is fully reconstructable and
challengeable without the audit log becoming a secondary disclosure channel for
the content it describes. The unsalted hash is correlatable by design, so
question frequency remains answerable without storing questions.

**Implementation:** `ahrag/audit/logger.py`.
**Verified by:** `tests/test_scenarios.py::TestAuditTrail` (7 tests, including
that evidence text never appears in a record).

**Prior-art risk: MEDIUM-HIGH.** Privacy-preserving audit logging, hashed
identifiers, and explainable-decision records are each well established across
security and ML-governance literature and products. The specific composition
applied to retrieval-route decisions may be narrower.

---

## 4. Prior art the author is already aware of

Any assessment must weigh at minimum the following. This list is from reading,
**not** from a professional search, and is certainly incomplete — in particular
it contains **no patent literature at all**, which is the most likely source of
blocking art.

| Work | Relevance | Why it matters here |
|---|---|---|
| Lewis et al. 2020, RAG | Foundational retrieve-then-generate | Establishes the base architecture |
| Karpukhin et al. 2020, DPR | Dense retrieval | The dense route; its rare-term weakness motivates R1 |
| Cormack et al. 2009, RRF | Rank fusion | R3/R4 use RRF directly and unmodified |
| **Jeong et al. 2024, Adaptive-RAG** | **Closest on adaptive routing** | Routes on predicted query complexity. M2 is defined as the delta from this. |
| Asai et al. 2024, Self-RAG | Reflection-token gating | Alternative approach to "should I retrieve / is this supported" |
| Yan et al. 2024, CRAG | Retrieval evaluator + corrective action | **Closest to M3.** Evaluates retrieval quality before generation. |
| Kalra et al. 2024, HyPA-RAG | Parameter-adaptive hybrid for legal/policy | Adaptive hybrid retrieval in a governed domain |
| **Zhao et al., R³AG** | **Closest on retriever routing** | Retriever routing using document assessment + downstream correctness |
| Cost-aware RAG routing work | Quality/cost/latency utility | **Closest to the utility formulation.** Same trade-off structure without governance terms. |
| **HONEYBEE (RBAC for vector DBs)** | **Closest to M1** | RBAC-aware partitioning in the vector database layer |
| ARES, RAGAS, RAGChecker | Evaluation frameworks | M5's methodology sits alongside these |
| Enterprise search products | ACL-filtered retrieval | Microsoft Graph/SharePoint security trimming, Elasticsearch document-level security, Glean, Coveo — all apply ACL pre-filtering as standard |

**The last row is the most important and the most often overlooked.** ACL-scoped
enterprise search is a mature commercial field with extensive patent activity.
Any claim touching M1 must be assessed against that portfolio, not against the
academic RAG literature.

---

## 5. Author's own assessment of relative strength

Ordered by the author's view of where differentiation is *most* likely to
survive scrutiny — while noting that none of these has been professionally
assessed and the honest expectation is that most will not.

| Rank | Mechanism | Assessment |
|---|---|---|
| 1 | **M2** — governance signals inside the routing utility | Narrowest delta from the closest art (Adaptive-RAG, R³AG, cost-aware routing). Most likely to be a genuine composition. |
| 2 | **M4** — non-destructive conflict disclosure | The mandatory non-suppression policy is a specific behavioural constraint, not just detection. |
| 3 | **M6** — decision auditability under minimisation | Composition may be narrow; components are all known. |
| 4 | **M3** — governance-scoped probe | Small delta over CRAG's retrieval evaluator. |
| 5 | **M5** — typed abstention + symmetric measurement | Methodological. Likely publishable, unlikely patentable. |
| 6 | **M1** — ACL as hard upstream constraint | **Likely unpatentable.** Standard enterprise-search practice. Valuable as a correctness property, not as IP. |

---

## 6. Enablement status

| Mechanism | Implemented | Tested | Reduced to practice |
|---|---|---|---|
| M1 | ✅ | ✅ 20 tests | ✅ on the demo corpus |
| M2 | ✅ | ✅ 30 tests | ✅ on the demo corpus |
| M3 | ✅ | ✅ 4 tests | ✅ on the demo corpus |
| M4 | ✅ | ✅ 5 tests | ✅ on the demo corpus |
| M5 | ✅ | ✅ symmetric metric tested | ✅ on the demo corpus |
| M6 | ✅ | ✅ 7 tests | ✅ on the demo corpus |

**All reduction to practice is on a 9-document, 60-chunk demo corpus with 32
labelled queries.** No mechanism has been validated at enterprise scale, and the
evaluation does not establish that any of them improves retrieval quality. See
`RESEARCH_LIMITATIONS.md`.

---

## 7. Disclosure status

- Source code, seeded corpus, and evaluation results are in this repository.
- Contributors should confirm whether this repository is public before assuming
  any grace period applies.
- The design derives from two internal documents (`docs/project_draft.docx`,
  `docs/literature_review.docx`) whose own disclosure status should be checked.

---

## 8. Explicit non-claims

For the avoidance of doubt, this document does **not** assert that:

- Any mechanism is novel, non-obvious, inventive, or patentable.
- Any mechanism is state of the art or outperforms any existing system.
- The measured evaluation results support any performance claim.
- Any freedom-to-operate position exists.
- The system is secure, production-ready, or compliant with any regulation.

All such determinations require professional review that has not been performed.
