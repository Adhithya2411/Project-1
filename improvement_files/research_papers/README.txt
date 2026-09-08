================================================================================
IMPROVEMENT FILES — MASTER INDEX
================================================================================
This folder contains all datasets, baseline code, evaluation tools, and
training templates needed to implement the improvements described in
improvement.txt (in the project root).

================================================================================
FOLDER STRUCTURE
================================================================================

improvement_files/
│
├── datasets/                        # Benchmark datasets for evaluation
│   ├── financebench_open_source.jsonl  ✅ 150 financial QA items (PatronusAI)
│   ├── financebench_hf.json            ✅ Same data via HuggingFace format
│   ├── financebench_repo/              ✅ Full FinanceBench repo with code
│   ├── policyqa_train.json             ✅ 17,056 policy QA examples
│   ├── hr_policies_qa.json             ✅ 644 HR policy QA examples
│   ├── CRAG_benchmark/                 ✅ Facebook's CRAG benchmark (KDD Cup 2024)
│   ├── download_hf_datasets.py         ✅ Script to re-download HF datasets
│   └── download_hotpotqa.py            ⏳ Script to download HotpotQA (7,405 multi-hop)
│
├── baseline_code/                   # Competing RAG systems to benchmark against
│   ├── Adaptive-RAG/                   ✅ Jeong et al., NAACL 2024
│   │   └── (T5-based query complexity classifier for routing)
│   ├── Self-RAG/                       ✅ Asai et al., ICLR 2024 (Oral)
│   │   └── (Self-reflective retrieval with reflection tokens)
│   └── Corrective-RAG/                ✅ Yan et al., 2024
│       └── (Retrieval evaluator + corrective web search)
│
├── evaluation_tools/                # Evaluation frameworks
│   ├── ragas/                          ✅ RAGAS framework (full repo)
│   │   └── (Faithfulness, answer relevancy, context precision/recall)
│   └── evaluate_with_ragas.py          ✅ Template to evaluate AHRAG with RAGAS
│
├── ml_router_training/              # Train a learned router to replace heuristics
│   └── train_router.py                 ✅ XGBoost/LightGBM router training template
│       └── Includes bootstrap CI and paired significance tests
│
├── corpus_sources/                  # Guides for expanding the corpus
│   └── (See CORPUS_EXPANSION_GUIDE below)
│
└── research_papers/                 # Key papers to cite and compete against
    └── PAPERS_LIST.txt                 ✅ All paper links, PDFs, and summaries


================================================================================
DOWNLOADED DATASETS SUMMARY
================================================================================

| Dataset          | Examples | Purpose for AHRAG                      | Status |
|------------------|----------|----------------------------------------|--------|
| FinanceBench     | 150      | Financial QA with evidence spans       | ✅     |
| PolicyQA         | 17,056   | Privacy policy reading comprehension   | ✅     |
| HR Policies QA   | 644      | HR-specific question answering         | ✅     |
| CRAG Benchmark   | 4,409    | Comprehensive RAG factual QA           | ✅     |
| HotpotQA         | 7,405    | Multi-hop QA (run download script)     | ⏳     |

Total available: ~29,664 labelled QA examples across domains.


================================================================================
BASELINE SYSTEMS CLONED
================================================================================

| System           | Paper               | Key Difference from AHRAG             |
|------------------|---------------------|---------------------------------------|
| Adaptive-RAG     | NAACL 2024          | Routes by query complexity only       |
| Self-RAG         | ICLR 2024 (Oral)    | Self-reflective tokens, no explicit routing |
| Corrective-RAG   | arXiv 2024          | Evaluates and corrects retrieved docs  |


================================================================================
WHAT TO DO NEXT (In Priority Order)
================================================================================

1. DOWNLOAD HotpotQA:
   python improvement_files/datasets/download_hotpotqa.py

2. EXPAND YOUR CORPUS:
   - Use PolicyQA's 115 privacy policies as real policy documents
   - Use HR Policies QA's 644 items for HR-specific evaluation
   - Use FinanceBench's 150 items for financial document evaluation
   - Collect 100+ additional documents from public sources (see improvement.txt §7)

3. TRAIN THE ML ROUTER:
   - pip install xgboost scikit-learn
   - Review and adapt improvement_files/ml_router_training/train_router.py
   - Generate training data by running all routes offline on expanded eval set

4. RUN RAGAS EVALUATION:
   - pip install ragas
   - Set OPENAI_API_KEY or ANTHROPIC_API_KEY
   - Adapt improvement_files/evaluation_tools/evaluate_with_ragas.py

5. COMPARE AGAINST BASELINES:
   - Study Adaptive-RAG's code in baseline_code/Adaptive-RAG/
   - Reproduce their routing on your dataset
   - Report side-by-side metrics with confidence intervals

6. DOWNLOAD RESEARCH PAPERS:
   - See improvement_files/research_papers/PAPERS_LIST.txt
   - Download all PDFs for citation in your thesis/paper


================================================================================
CORPUS EXPANSION GUIDE (improvement.txt §7 — Phase 1)
================================================================================

Collect documents from these PUBLIC sources:

HR/POLICY DOCUMENTS:
  - OpenGov Foundation: https://github.com/opengovfoundation/hr-resources (CC0)
  - Kaggle HR Policy PDFs: https://www.kaggle.com/datasets/harekalrajesh/hr-policy-docs-pdf
  - US Government HR policies: https://www.opm.gov/policy-data-oversight/
  - University handbooks: Search "[university name] employee handbook PDF"

TECHNICAL RUNBOOKS:
  - Kubernetes troubleshooting: https://kubernetes.io/docs/tasks/debug/
  - AWS incident response: https://docs.aws.amazon.com/whitepapers/latest/aws-security-incident-response-guide/
  - Open-source project runbooks: Search GitHub for "runbook" or "playbook"

FINANCIAL REPORTS:
  - SEC EDGAR (10-K filings): https://www.sec.gov/cgi-bin/browse-edgar
  - Company annual reports: Most are freely available on company websites

PROJECT REPORTS:
  - Open-source project reports: Apache, Linux Foundation, CNCF project reports
  - Government project reports: data.gov, data.gov.uk

Target: 500+ documents, 5,000+ chunks, at least 10 supersession chains.

================================================================================
