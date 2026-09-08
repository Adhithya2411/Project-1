# AHRAG (Adaptive Hybrid Retrieval-Augmented Generation)
## Project Context and Status Document

This document provides a comprehensive overview of the AHRAG project, detailing its architecture, the core idea, the purpose of the improvement files, and the current state of the codebase. It is designed to serve as a high-fidelity context injection for future AI sessions or developers joining the project.

---

## 1. Project Overview & Core Idea

**AHRAG** is an advanced, **governance-aware** Retrieval-Augmented Generation (RAG) system. While traditional RAG systems simply embed text and perform nearest-neighbor searches, AHRAG is built for enterprise environments where data privacy, access control lists (ACLs), data freshness, and conflict disclosure are critical.

### Key Innovations:
1. **Governance-Awareness:** The system actively respects user roles (ACLs). If a user queries for financial data but lacks the `finance` role, the system will explicitly withhold that chunk and log the withholding in an audit trace.
2. **Adaptive Routing:** Not all queries require expensive multi-hop vector searches. AHRAG categorizes queries and routes them to the cheapest/most effective retrieval mechanism (e.g., Sparse BM25, Dense Vector, Hybrid RRF, or Multi-Hop Iterative).
3. **Evidence Sufficiency & Abstention:** If retrieved evidence is contradictory or insufficient, the system abstains from answering rather than hallucinating, and can ask clarifying questions.

---

## 2. Core Pipeline Architecture

The core operational code of the system lives in the `ahrag/` directory.

### Key Files:
* `ahrag/pipeline.py`: Contains the `AHRAGEngine`, which represents the 13-stage lifecycle of a query. The primary entry point for queries is `engine.answer()`. It orchestrates ACL scoping, probe generation, feature extraction, routing, retrieval, evidence packing, and generation.
* `ahrag/routing/router.py`: Implements the routing logic. It evaluates the query's complexity and constraints to pick between routes (R0: Abstain, R1: Sparse, R2: Dense, R3: Hybrid, R4: Iterative).
* `ahrag/models.py`: Contains all Pydantic schemas defining the domain (e.g., `Chunk`, `Document`, `AnswerResult`, `AuditRecord`, `EvidenceItem`).
* `ahrag/db.py`: The SQLite-backed database layer that stores documents, chunks, and ACL metadata.
* `ahrag/eval/systems.py` & `ahrag/evaluate.py`: The evaluation harness. It defines 6 different system configurations (Baselines B1-B5, and the proposed AHRAG P1) to run head-to-head comparisons.

---

## 3. The `improvement_files/` Modules

The `improvement_files/` directory contains tools and scripts designed to elevate AHRAG from a proof-of-concept into a statistically validated, machine-learning-driven production model. 

Initially, these files were empty templates, broken scaffolds, and relied heavily on "mock data" (e.g., hardcoded string responses, Lorem Ipsum documents, skipped ML training). **As of the current status, all of these files have been completely rewritten to interface directly with the real AHRAG engine and process genuine data.**

### 3.1. Machine Learning Router (`improvement_files/ml_router_training/`)
* **File:** `[train_router.py](file:///e:/SEM-7/Project/Project-1/improvement_files/ml_router_training/train_router.py)`
* **Purpose:** Upgrades the rule-based governance-aware router to a Gradient-Boosted Classifier (`xgboost`). 
* **How it works:** It boots up the `AHRAGEngine`, passes the training datasets through the engine to extract **23 real mathematical features** (like `restricted_fraction`, `hop_signal`, `lexical_specificity`), determines the "gold" route that retrieves the correct chunks cheapest, and trains an XGBoost model.
* **Output:** Saves a trained model to `artifacts/xgboost_router.json`.

### 3.2. Dataset Integration (`improvement_files/datasets/`)
* **File:** `[integrate_datasets.py](file:///e:/SEM-7/Project/Project-1/improvement_files/datasets/integrate_datasets.py)`
* **Purpose:** Ingests external, real-world benchmark datasets into the AHRAG corpus to drastically expand the evaluation scope.
* **How it works:** It parses thousands of raw JSON records from **FinanceBench**, **PolicyQA**, and **HotpotQA**. It maps them into AHRAG `documents` (with simulated ACL roles and metadata) and `eval_items` (queries mapped to specific gold-chunk IDs).
* **Output:** Generates `manifest.yaml`, `eval_set.yaml`, and text files in `integrated/`.

### 3.3. Public Corpus Collection (`improvement_files/corpus_sources/`)
* **File:** `[collect_public_corpus.py](file:///e:/SEM-7/Project/Project-1/improvement_files/corpus_sources/collect_public_corpus.py)`
* **Purpose:** Fetches genuine policy and governance documents from the web to serve as the system's foundational knowledge base.
* **How it works:** It scrapes open-source repositories and government databases for documents like Kubernetes Troubleshooting Runbooks, Node.js Code of Conducts, and NIST Cybersecurity Guidelines.
* **Output:** Saves real text files to `collected/corpus/` and creates a `manifest.yaml` for AHRAG ingestion.

### 3.4. RAGAS Evaluation (`improvement_files/evaluation_tools/`)
* **File:** `[evaluate_with_ragas.py](file:///e:/SEM-7/Project/Project-1/improvement_files/evaluation_tools/evaluate_with_ragas.py)`
* **Purpose:** Integrates AHRAG with the industry-standard **RAGAS** framework to evaluate the LLM's generated text quality.
* **How it works:** It loops through AHRAG's evaluation set, runs the full `engine.answer()` pipeline, and collects the actual generated text and retrieved contexts. It then uses RAGAS (via an OpenAI LLM judge) to score metrics like **Faithfulness**, **Answer Relevancy**, and **Context Precision**.
* **Output:** Produces `data/ragas_input.json` and (if an API key is provided) a `_ragas_scores.json` report.

### 3.5. Baseline Comparisons (`improvement_files/baseline_code/`)
* **File:** `[compare_baselines.py](file:///e:/SEM-7/Project/Project-1/improvement_files/baseline_code/compare_baselines.py)`
* **Purpose:** Proves mathematically that AHRAG's proposed routing (P1) is superior to standard RAG implementations.
* **How it works:** It runs every single evaluation query through 6 different systems:
  - **B1**: Fixed Sparse BM25
  - **B2**: Fixed Dense Embedding
  - **B3**: Fixed Hybrid RRF
  - **B4**: Fixed Multi-Hop Iterative
  - **B5**: Complexity-Only Adaptive Router (ignores ACLs/governance)
  - **P1**: AHRAG Governance-Aware Router
  It then calculates Recall@5 and ACL Violation rates, and computes **Paired Bootstrap Significance Tests** to determine if P1's improvements are statistically significant.

---

## 4. Current Project Status

1. **Phase 1: Discovery & Audit (COMPLETED)**
   - Initial codebase review was performed. The lack of functional code in `improvement_files` was identified.
2. **Phase 2: Implementation (COMPLETED)**
   - All mock scripts were replaced with functional Python scripts interacting with the real engine.
   - Missing dependencies (`xgboost`, `scikit-learn`, `pyyaml`, `pydantic`) were installed.
   - Windows terminal encoding bugs (`UnicodeEncodeError` on characters like `→`, `α`, `✓`) were resolved to ensure seamless execution on Windows.
   - The codebase is stable.
   - The user can execute dataset integrations, ML training, baseline comparisons, and RAGAS evaluations right now using the scripts in `improvement_files`.
   - The AI is primed to assist with further optimizations, adding new external datasets, tuning the XGBoost hyperparameters, or modifying the core `ahrag` engine architecture if requested.

---

## 5. Developer Execution Guide

If an AI or developer needs to re-run the pipeline from scratch, follow this execution order:

1. **Ingest Datasets:**
   ```bash
   python improvement_files/datasets/integrate_datasets.py
   ```
2. **Collect Public Corpus:**
   ```bash
   python improvement_files/corpus_sources/collect_public_corpus.py
   ```
3. **Train the ML Router:**
   ```bash
   python improvement_files/ml_router_training/train_router.py
   ```
4. **Evaluate with RAGAS:**
   ```bash
   python improvement_files/evaluation_tools/evaluate_with_ragas.py --run
   ```
5. **Run Baseline Statistical Comparisons:**
   ```bash
   python improvement_files/baseline_code/compare_baselines.py
   ```

## 6. Key API Concepts for AI Context
- **`AHRAGEngine.answer(query, user_id)`**: The absolute core function. Always use this to get a response. It handles everything (ACLs, routing, generation).
- **`Database.get_chunks()`**: Used to retrieve raw data for evaluation ground-truths.
- **ACL Scoping**: Never bypass `engine.acl.scope_for(user)`. AHRAG's main feature is that it enforces these scopes strictly.
