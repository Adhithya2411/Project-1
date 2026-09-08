# Results — charts and figures

Presentation-ready assets generated from the experiment reports in
`data/reports/`. Everything here is derived; nothing is hand-entered.

Regenerate after any experiment re-run:

```bash
python improvement_files/evaluation_tools/make_result_charts.py
```

`RESULTS.txt` carries every number as plain text, so the same figures are
available without opening an image — useful for pasting into a document, and
for the case where a chart is printed in greyscale.

---

## Files

| File | What it shows | Use it for |
|---|---|---|
| `01_system_recall_at_5.png` | Recall@5 for all 8 systems with 95% bootstrap CIs | The headline comparison slide |
| `02_ablation_recall.png` | Effect on Recall@5 of disabling each mechanism | Which components matter |
| `03_ablation_abstention.png` | Effect on abstention appropriateness | The strongest ablation result |
| `04_embedding_backends.png` | LSA versus a neural encoder | The §4 "low-hanging fruit" result |
| `05_spis_lambda_frontier.png` | What non-interference costs, as λ varies | The novel contribution's trade-off |
| `06_spis_interference.png` | Interference from unreadable documents | The security finding |
| `07_router_learning_curve.png` | Validation accuracy versus training-set size | That more labels help |
| `RESULTS.txt` | Every number, as text | Pasting; greyscale printing |

---

## What each chart is actually saying

Read these before putting a figure on a slide — several of them do **not** say
what a quick glance suggests.

### 01 — System comparison

The learned router (P2) is highest on Recall@5, and the hand-tuned governance
router (P1) is *below* four of the six baselines. The architecture holds up; the
hand-picked utility weights do not. The λ grid search says the same thing
independently — the shipped `0.09/0.35/0.55` ranks 7th of 27.

**Do not present P2's number as a clean held-out win.** P2 was trained on
offline route labels derived from 40% of these same queries, and no other
system had any training data. `data/reports/baselines_heldout.json`, when
present, is the same comparison restricted to P2's untouched test split; that
is the figure to quote for a fair claim.

Also worth saying on the slide: **zero ACL violations for all eight systems**.
That is the intended result, not a differentiator — ACL enforcement sits
upstream of routing, so no routing policy can break it. The metric exists to
verify the invariant, and at 965 queries across 7 principals it holds.

### 02 and 03 — Ablations

The dominant bar is `− probe`: removing the governance-scoped probe drops
Recall@5 from 0.507 to 0.013 and abstention appropriateness by 0.698
(Cohen's *d* = −1.52). The router falls back to abstaining on 96% of queries,
because without probe evidence the quality model cannot tell that answering is
worthwhile. This is the strongest single result in the project.

The second-largest bar is the uncomfortable one. Disabling the **evidence
sufficiency gate** *improves* abstention appropriateness by +0.105
(p < 0.0001). The gate is mis-calibrated at scale: its thresholds are absolute
rerank scores fitted to a 60-chunk corpus, and at 7,082 chunks they cause
inappropriate abstentions. That is a defect the ablation found, and it is not
yet fixed.

Several mechanisms show no measurable effect at all. That is a real finding
about this corpus, not missing data.

### 04 — Embedding backends

The neural encoder wins by +0.0215 Recall@5 at p = 0.002. Both halves matter:
the same comparison on the 9-document demo corpus gave +0.040 at p = 0.52 —
indistinguishable from noise. The gain is a third the size and *reliably
measurable*, which is the clearest demonstration in the project of why the
evaluation set had to grow. Cohen's *d* ≈ 0.10 is negligible by convention, so
the honest phrasing is "real but small", not "neural embeddings fix retrieval".

### 05 and 06 — Scope-Pure Index Specialisation

06 is the finding: with globally-fitted retrieval statistics, deleting
documents a principal **cannot read** changed that principal's route on a
third of queries. No restricted text is ever returned, so this is not an
access-control violation — it is an information-flow violation. Access control
constrains outputs; non-interference constrains dependence. AHRAG enforced the
first and not the second.

05 is the price: at λ = 0 the index is provably non-interfering and costs
0.0040 Recall@5 (p = 0.027, *d* = −0.071) — about one item in 250. The cost
does not disappear on a wide ACL lattice, so **the contribution is the security
property alone, not a retrieval gain.** That was predicted in writing before the
run; see `FINDINGS.md` §5.

λ governs the sparse channel only, so this is half a frontier, not a complete
one.

### 07 — Learning curve

Validation route accuracy rises from 0.695 at 38 training queries to 0.813 at
386, then flattens. Labels are derived by executing all five routes on every
query and taking the cheapest one that achieved the best measured recall — not
from any annotator's intuition about which route ought to win.

---

## Caveats that apply to every figure here

1. **Check which corpus.** Each chart's subtitle names it. The demo corpus
   (9 documents) and the benchmark corpus (6,139) give very different absolute
   numbers; the benchmark ones are lower because retrieval there is genuinely
   hard.
2. **Absolute recall ~0.5 is dominated by one stratum.** `policy_comprehension`
   is 300 of 965 queries and sits at ~0.16 for every system — short
   reading-comprehension questions over privacy-policy paragraphs, where
   finding the right paragraph among 7,082 chunks is hard. That is a difficulty
   result, not a defect.
3. **Freshness reads 1.000 for all eight systems.** It is now genuinely
   exercised — an earlier version of the generated freshness queries was
   unanswerable and every system scored 0.000, which made the metric silently
   report perfection. It no longer discriminates *between* systems, so it
   supports "freshness handling works" and not "ours is better".
4. **Two strata have no recall by construction.** `permission_boundary` and
   `unanswerable` have no gold chunks; they are scored by abstention
   appropriateness. A dash means undefined, not zero.
5. **No inter-annotator agreement figure exists.** The labels are
   programmatic. `improvement_files/evaluation_tools/annotation_agreement.py`
   provides the tooling and the statistics, but a second *human* annotator is
   required and none has run.

`FINDINGS.md` is the full record, including the results that came out negative.
