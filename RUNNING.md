# Running and Testing AHRAG

A practical guide: install it, start the web app, and verify by hand that it
does what the project claims. Every command and every expected output below was
executed on this machine — where a result is surprising or a step is easy to get
wrong, that is called out rather than left for you to discover.

Companion documents: `FINDINGS.md` (all measured results), `context.md`
(architecture and status), `README.md` (design walkthrough).

---

## 1. Install

Python 3.11+ (developed on 3.12; this machine ran 3.14).

```bash
cd Project-1
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

That is enough to run everything: the system is fully offline by default, with
no model downloads and no API keys.

### Optional extras

Each is genuinely optional and the system degrades cleanly without it.

```bash
# Learned router (P2) and the Adaptive-RAG baseline (B6)
brew install libomp                 # macOS only, required by xgboost
pip install xgboost scikit-learn

# Neural embeddings + cross-encoder reranker
pip install sentence-transformers

# LLM-judge metrics
pip install ragas datasets
export OPENAI_API_KEY=sk-...
```

> **macOS: install `libomp` before `xgboost`.** Without it `import xgboost`
> fails with `Library not loaded: @rpath/libomp.dylib`.

---

## 2. Start the web app

The app is two processes. The UI works with or without the API — it probes
`/api/health` and falls back to an in-process engine, saying which in the
sidebar.

### Terminal 1 — the API

```bash
source .venv/bin/activate
export AHRAG_EMBEDDING_BACKEND=lsa
export AHRAG_RERANKER=lexical
python -m uvicorn "ahrag.api.app:create_app" --factory --port 8000
```

### Terminal 2 — the UI

```bash
source .venv/bin/activate
export AHRAG_EMBEDDING_BACKEND=lsa
export AHRAG_RERANKER=lexical
streamlit run ahrag/ui/app.py
```

Then open **http://localhost:8501**.

> **Why export those two variables?** The default is `auto`, which silently
> switches to a neural encoder the moment `sentence-transformers` is installed.
> That changes the numbers you see and, on Python 3.14, can abort the process.
> Pin them for demos so runs are reproducible; unset them deliberately when you
> *want* to compare backends.

### Confirm it came up

```bash
curl -s http://localhost:8000/api/health | python -m json.tool
```

Expected — the shape matters more than the exact values:

```json
{
  "status": "ok",
  "version": "0.3.0",
  "backends": {
    "sparse": "rank-bm25/BM25Okapi",
    "embedder": "lsa",
    "reranker": "lexical",
    "generator": "extractive-deterministic",
    "router": "ahrag-governance-aware",
    "documents": 9,
    "chunks": 60,
    "index_specialisation": { "enabled": false, "fully_pure": false }
  }
}
```

`documents: 9` is correct on a first run — that is the packaged demo corpus.
See §6 for loading the 6,139-document benchmark corpus.

---

## 3. The four pages

| Page | What it is for |
|---|---|
| **Ask** | Pick a user, ask a question, inspect the route decision, evidence, citations, conflicts and audit trail. |
| **Corpus & ingestion** | Browse documents and their ACL roles; upload a new document with roles and an effective date. |
| **Audit log** | Every recorded decision, with route distribution. Note what is *not* stored. |
| **Evaluation** | Run the 8-system comparison from the browser. |

The sidebar has the two controls that matter: **Corpus** and **Signed in as**.
Changing the user without changing the question is the single most informative
thing you can do in this UI.

---

## 4. Test it by hand — six checks

These are the behaviours the project claims. Each is a click-through with the
result I actually observed.

### Check 1 — The same question, two users (the headline)

On **Ask**, ask this *twice*, changing only the user:

> `What is the Q3 2026 revenue forecast?`

| Signed in as | Expected |
|---|---|
| **Dan Oyelaran — Finance Analyst** | **Answers**, citing `doc-fin-q3-forecast`. |
| **Erin Vasquez — External Contractor** | Red banner: **Abstained — reason `no_authorised_evidence`**. Route **R0**. **Zero** evidence rows. |

> **Do not expect a specific route for the finance user.** It is R2 or R3
> depending on index state, and it legitimately shifts when the corpus changes
> — uploading a document is enough to move it, because BM25 statistics are
> fitted over the corpus. What is invariant, and what the check is actually
> testing, is that the finance user *answers from the restricted document* and
> the contractor *abstains with nothing*. The contractor's **R0** is not
> incidental: it is the hard governance constraint firing.

Now look carefully at the contractor's screen. The refusal says *"I can't answer
that from the sources you're authorised to read."* It does **not** say a
restricted document exists, and `doc-fin-q3-forecast` appears nowhere on the
page. An abstention that leaks the existence of the file it is protecting has
failed at its job — verified absent.

### Check 2 — Conflicting policy versions are disclosed

Signed in as **Carol Bianchi — HR Business Partner**:

> `What is the current annual leave entitlement now?`

Expected: route **R3**, and a **conflicts** section listing **two** conflicts:

- a **version** conflict — two versions of the Annual Leave policy are both in
  the evidence;
- a **numeric** conflict — *"Authorised sources state different values for
  'annual leave': 22 day … 26 day …"*

The answer leads with **26 days** (the current version), the preference is
explained, and a freshness warning states the v1.4 document *"is shown because
it is still authorised and relevant, not because it is current."* The
disagreement is surfaced, not silently resolved.

### Check 3 — Scope changes evidence quality, not just visibility

Ask the same question as two different users:

> `What does error ERR-5041 mean?`

- **Alice Nowak (engineering)** — answers with `doc-eng-payments-runbook`
  among its sources: the authoritative runbook for that error code.
- **Erin Vasquez (contractor)** — also answers, but from
  `doc-pmo-atlas-q2` **alone**, a *project status report* that happens to
  mention the code in passing.

Both are correct and neither leaks: the runbook is `engineering`-only and stays
withheld, while the status report grants `employee`. This is the interesting
case — authorisation does not merely hide rows, it changes which source you get
an answer from.

### Check 4 — Abstention, and where it is unreliable

> `Which vendor supplies the executive submarine fleet?`

Expected: abstains. Nothing in the corpus supports it.

Now try:

> `What is the company policy on interplanetary business travel?`

**It answers.** This is a real weakness, not a demo bug: "policy" and "business
travel" match the real travel-expense policy strongly enough (probe confidence
0.72) to clear both the probe floor and the sufficiency gate. Four of five
obviously-unsupported questions behave this way. See `FINDINGS.md` §4.2 — the
ablation study independently found the sufficiency gate is mis-calibrated.

Showing this is more useful than hiding it, so the demonstration test suite
asserts it explicitly.

### Check 5 — The audit log records the decision, not the content

Ask anything, then open **Audit log**. Each record has `route`,
`router_confidence`, `features`, `candidate_utilities`, `documents_considered`,
`documents_used`, `citations`, `acl_pool_size`, `acl_withheld_count`, timings.

What it does **not** have: `query_text` is `null`, and no evidence text appears
anywhere. Only `query_hash` is stored. A route decision is fully
reconstructable without retaining the source material.

```bash
curl -s "http://localhost:8000/api/audit?limit=1" | python -m json.tool | head -30
```

### Check 6 — Upload a document and query it immediately

On **Corpus & ingestion**, upload any `.md`/`.txt` with `acl_roles = employee`.
Then ask a question about its content. It is retrievable at once — ingestion
rebuilds the indexes synchronously.

Try uploading **without** ACL roles: the response carries a `warning` and the
document is restricted rather than public. Failing closed is the intended
default.

---

## 5. Test the API directly

```bash
# every endpoint
for p in /api/health /api/users /api/roles /api/documents /api/eval-runs "/api/audit?limit=1"; do
  printf "%-26s %s\n" "$p" "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:8000$p")"
done
```

All six return `200`. The full route list is `GET /api/health`,
`GET /api/users`, `GET /api/roles`, `GET /api/documents`, `GET /api/eval-runs`,
`GET|DELETE /api/audit`, `POST /api/query`, `POST /api/ingest`, `POST /api/seed`.
Interactive docs: **http://localhost:8000/docs**.

```bash
# the ACL boundary, as two calls
curl -s -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the Q3 2026 revenue forecast?","user_id":"dan.finance"}' \
  | python -c "import sys,json; r=json.load(sys.stdin)['result']; \
print('route', r['decision']['route'], '| abstained', r['abstained'], \
'| docs', sorted({e['doc_id'] for e in r['evidence']}))"

curl -s -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the Q3 2026 revenue forecast?","user_id":"erin.contractor"}' \
  | python -c "import sys,json; r=json.load(sys.stdin)['result']; \
print('route', r['decision']['route'], '| abstained', r['abstained'], \
'| leak', 'doc-fin-q3-forecast' in json.dumps(r))"
```

Expected (observed on a freshly seeded corpus):

```
route R3 | abstained False | docs ['doc-fin-q3-forecast', 'doc-pmo-atlas-q2']
route R0 | abstained True  | leak False
```

The finance user's route and the exact document set vary with index state; the
invariants are `abstained False` **with `doc-fin-q3-forecast` present** on the
first line, and `abstained True` **with `leak False`** on the second. If you
have uploaded documents during a session, reset before comparing:

```bash
curl -s -X POST http://localhost:8000/api/seed \
  -H 'Content-Type: application/json' -d '{"reset": true}'
```

> **The response is wrapped.** Everything is under a top-level `result` key —
> `json['result']['decision']['route']`, not `json['decision']['route']`.

Error paths: unknown `user_id` → **404**; empty `query` → **422**.

---

## 6. Using the 6,139-document benchmark corpus

The demo corpus exists so the app runs offline with no setup. Every published
number in `FINDINGS.md` comes from the benchmark corpus instead.

```bash
# build it once (~4 min; cached under data/corpus-cache afterwards)
python improvement_files/datasets/integrate_datasets.py
```

Produces 6,139 documents / 7,082 chunks / 965 labelled queries, with a 7-class
ACL lattice and 15 supersession chains. Of those documents, **249 are
enterprise-shaped** (policy, finance, runbook) and 5,890 are single-paragraph
reference articles included so first-stage retrieval is genuinely selective —
quote the 249 when comparing against a "number of enterprise documents"
target. Then either point the UI's **Corpus**
selector at it, or run the experiments against it:

```bash
python improvement_files/baseline_code/compare_baselines.py --integrated --by-type
python improvement_files/baseline_code/ablation_study.py --integrated
python improvement_files/evaluation_tools/compare_embeddings.py --integrated
python improvement_files/evaluation_tools/measure_specialisation.py --integrated
python improvement_files/ml_router_training/train_router.py --integrated
```

Every script also takes `--manifest` / `--eval-set` for an arbitrary corpus, and
**validates that the evaluation labels resolve against the corpus loaded**. That
check exists because its absence previously let 83% of gold chunk IDs point at
the wrong text while every metric still produced a plausible-looking number.

---

## 7. Run the tests

```bash
python -m pytest tests/ -q                        # 290 tests, ~5 s
```

The guided tour is the most useful one to run by hand — ten scenarios that each
print their evidence:

```bash
python -m pytest tests/test_demonstration.py -s -q
```

```bash
# one scenario at a time
python -m pytest tests/test_demonstration.py -s -k acl_boundary
python -m pytest tests/test_demonstration.py -s -k conflict
python -m pytest tests/test_demonstration.py -s -k non_interference

# the novel contribution's proof obligations
python -m pytest tests/test_noninterference.py -q   # 26 tests
```

---

## 8. Troubleshooting

Each of these cost real time to diagnose; the cause is not guessable from the
symptom.

| Symptom | Cause and fix |
|---|---|
| Process dies with **no traceback**, stdout empty | `xgboost` and `torch` each bundle an OpenMP runtime and loading torch *first* aborts the interpreter on macOS. `ahrag/_openmp.py` imports xgboost first; do not remove its import from `ahrag/__init__.py`. |
| **Segfault partway through pytest** | `sentence-transformers` is installed and `AHRAG_EMBEDDING_BACKEND=auto` loads torch inside session-scoped fixtures. `tests/conftest.py` pins the offline backends; keep it that way. |
| Comparing **two transformer models** in one run dies | One sentence-transformer per process on this platform. Run a separate process per backend. |
| `XGBoostError: libxgboost.dylib could not be loaded` | `brew install libomp`. |
| **Leaked semaphore** warning during training | xgboost's loky workers on Python 3.14. Already pinned to `n_jobs=1`; the warning at exit is harmless. |
| Numbers differ from `FINDINGS.md` | Check which corpus. `documents: 9` in `/api/health` means the demo corpus; the published figures use `--integrated`. |
| UI says it is using an **in-process engine** | The API is not reachable. Fine for demos; start the API if you want them coupled. |
| Metric reads `n/a` in a script you wrote | Per-row keys from `run_item` are not the aggregate names. The row key is `abstention_appropriate` (bool per query); the aggregate is `abstention_appropriateness` (its mean). Reading the aggregate off a row yields `None` for every item, silently. |
| Corpus rebuild seems stuck | First ingest of the benchmark corpus is ~4 minutes. Subsequent runs load the cache in ~14 seconds. Force a rebuild with `--rebuild-cache`. |

---

## 9. Fastest possible demo

If you have two minutes and want to show the point of the project:

```bash
source .venv/bin/activate
export AHRAG_EMBEDDING_BACKEND=lsa AHRAG_RERANKER=lexical
streamlit run ahrag/ui/app.py
```

On the **Ask** page, ask `What is the Q3 2026 revenue forecast?` as
**Dan Oyelaran (Finance)** — it answers from the restricted forecast. Change
only the user to **Erin Vasquez (Contractor)** and ask again — it abstains, with
no evidence and no hint that the document exists.

Then ask `What is the current annual leave entitlement now?` as
**Carol Bianchi (HR)** and open the conflicts section: the system reports that
two authorised sources say 22 days and 26 days, answers with 26, and explains
why it preferred that one.

That is the whole thesis in two questions: the route and the answer are
functions of *who is asking*, and disagreement between sources is disclosed
rather than hidden.
