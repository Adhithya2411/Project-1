"""Streamlit frontend for AHRAG.

Runs against the FastAPI backend when it is reachable, and falls back to an
in-process engine otherwise, so the UI is usable with a single command. The
active mode is shown in the sidebar — a reviewer should never be unsure whether
they are looking at API-served or in-process results.

Run with::

    streamlit run ahrag/ui/app.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Allow `streamlit run ahrag/ui/app.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ahrag import __version__  # noqa: E402
from ahrag.config import RouterConfig, Settings  # noqa: E402
from ahrag.db import Database  # noqa: E402
from ahrag.models import DocType  # noqa: E402
from ahrag.eval.reports import load_reports  # noqa: E402
from ahrag.pipeline import AHRAGEngine  # noqa: E402

ROUTE_HELP = {
    "R0": "Clarification or evidence-based abstention. Never answers from model memory.",
    "R1": "Sparse BM25 — exact identifiers, codes, clauses, rare terms.",
    "R2": "Dense vectors — semantic and conceptual questions.",
    "R3": "Hybrid sparse+dense fused with Reciprocal Rank Fusion.",
    "R4": "Decomposed iterative hybrid — comparison, temporal, multi-hop.",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Corpora the UI can seed. Only those whose manifest is actually present are
# offered, so a fresh clone shows the demo corpus alone rather than listing
# options that fail on click — the benchmark corpora are build products, not
# checked-in fixtures in every deployment.
_CANDIDATE_CORPORA: list[tuple[str, Path | None]] = [
    ("Demo corpus (9 documents)", None),
    (
        "FinanceBench subset (30 documents)",
        _REPO_ROOT / "improvement_files" / "datasets" / "financebench_subset"
        / "manifest.yaml",
    ),
    (
        "Public governance corpus (73 documents)",
        _REPO_ROOT / "improvement_files" / "corpus_sources" / "collected"
        / "manifest.yaml",
    ),
    (
        "Benchmark corpus (6,139 documents — slow first load)",
        _REPO_ROOT / "improvement_files" / "datasets" / "integrated"
        / "manifest.yaml",
    ),
]

CORPUS_OPTIONS: dict[str, str | None] = {
    label: (str(path) if path is not None else None)
    for label, path in _CANDIDATE_CORPORA
    if path is None or path.exists()
}

# Corpora large enough that seeding them from a browser click needs a warning
# rather than a spinner. Ingesting the benchmark corpus takes several minutes,
# and Streamlit gives no progress signal during a synchronous call.
SLOW_CORPUS_DOC_THRESHOLD = 500
SLOW_CORPORA = {
    label
    for label, path in _CANDIDATE_CORPORA
    if path is not None and path.exists() and path.stat().st_size > 100_000
}


# ---------------------------------------------------------------------------
# Backend access
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Starting AHRAG engine…")
def get_local_engine(manifest_path: str | None = None) -> AHRAGEngine:
    """Build and cache an in-process engine for the selected corpus."""
    settings = Settings()
    config = RouterConfig.load(settings.router_config)
    db = Database(settings.db_path)
    engine = AHRAGEngine(settings=settings, config=config, db=db)
    engine.seed(reset=True, manifest_path=manifest_path)
    return engine


def api_base() -> str | None:
    """Return the API base URL if the backend is reachable, else None."""
    import httpx

    url = Settings().api_base_url.rstrip("/")
    try:
        response = httpx.get(f"{url}/api/health", timeout=1.5)
        if response.status_code == 200:
            return url
    except Exception:  # noqa: BLE001 - unreachable API is the expected case
        return None
    return None


def call_api(base: str, method: str, path: str, **kwargs: Any) -> Any:
    """Make an API call and raise a readable Streamlit error on failure."""
    import httpx

    try:
        response = httpx.request(method, f"{base}{path}", timeout=60.0, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:  # noqa: BLE001
            detail = exc.response.text[:300]
        st.error(f"API error {exc.response.status_code}: {detail}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not reach the API: {exc}")
        st.stop()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_route_badge(route: str, confidence: float) -> None:
    """Render the chosen route as a coloured banner."""
    colour = {"R0": "🟥", "R1": "🟦", "R2": "🟩", "R3": "🟪", "R4": "🟧"}.get(route, "⬜")
    st.markdown(
        f"### {colour} Route **{route}** &nbsp;·&nbsp; confidence {confidence:.2f}"
    )
    st.caption(ROUTE_HELP.get(route, ""))


def render_why_panel(decision: dict[str, Any]) -> None:
    """Render the 'Why this route?' explanation panel."""
    st.subheader("Why this route?")

    cols = st.columns(4)
    cols[0].metric("Est. latency", f"{decision['estimated_latency_s']:.3f} s")
    cols[1].metric("Est. cost", f"${decision['estimated_cost_usd']:.6f}")
    cols[2].metric("Est. evidence risk", f"{decision['estimated_evidence_risk']:.3f}")
    cols[3].metric("Fallback", decision["fallback_route"])

    if decision.get("hard_constraints_applied"):
        st.warning(
            "**Hard governance constraints applied before any utility was "
            "compared:**\n\n"
            + "\n".join(f"- {c}" for c in decision["hard_constraints_applied"])
        )

    st.markdown("**Reasons**")
    for reason in decision["reasons"]:
        st.markdown(f"- {reason}")

    st.markdown("**Candidate route utilities**")
    st.caption(
        "U = expected evidence quality − λ_latency·latency − λ_cost·cost − "
        "λ_risk·risk. Routes marked *not admissible* were excluded by a hard "
        "governance constraint before utility was considered — no latency or "
        "cost saving can reinstate them."
    )
    rows = []
    for util in decision["utilities"]:
        rows.append(
            {
                "route": util["route"],
                "admissible": "yes" if util["admissible"] else "NO",
                "why not": util.get("rejection_reason") or "—",
                "quality": round(util["expected_evidence_quality"], 3),
                "−latency": round(util["latency_penalty"], 3),
                "−cost": round(util["cost_penalty"], 3),
                "−risk": round(util["risk_penalty"], 3),
                "utility": round(util["utility"], 3),
                "chosen": "◀" if util["route"] == decision["route"] else "",
            }
        )
    frame = pd.DataFrame(rows).sort_values("utility", ascending=False)
    st.dataframe(frame, width="stretch", hide_index=True)

    with st.expander("Router feature values"):
        features = decision["features"]
        display = [
            {"feature": key, "value": str(value)}
            for key, value in features.items()
            if key not in {"intent_scores", "user_roles", "identifiers_found"}
        ]
        display.append(
            {"feature": "identifiers_found", "value": ", ".join(features.get("identifiers_found", [])) or "—"}
        )
        st.dataframe(pd.DataFrame(display), width="stretch", hide_index=True)
        st.caption(f"Router version: `{decision['router_version']}`")


def render_evidence(evidence: list[dict[str, Any]]) -> None:
    """Render the evidence table and expandable passages."""
    if not evidence:
        st.info("No evidence was packed for this query.")
        return

    frame = pd.DataFrame(
        [
            {
                "chunk_id": e["chunk_id"],
                "source": e["title"],
                "type": e["doc_type"],
                "score": round(e["score"], 4),
                "retriever": e["retriever"],
                "version": e["version"],
                "effective": e["effective_date"],
                "authority": f"{e['authority_score']}/5",
                "ACL": e["acl_status"],
                "superseded": "yes" if e["is_superseded"] else "no",
                "subquery": e.get("subquery") or "—",
            }
            for e in evidence
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption(
        "Every row is authorised for the selected user. Unauthorised chunks are "
        "removed before retrieval, so they cannot appear here even as a filtered row."
    )

    for item in evidence:
        label = (
            f"[{item['chunk_id']}] — {item['title']} v{item['version']} "
            f"(authority {item['authority_score']}/5)"
        )
        with st.expander(label):
            st.caption(
                f"owner: {item['owner']} · effective {item['effective_date']} · "
                f"readable by: {', '.join(item['acl_roles'])} · source: "
                f"`{item['source_uri']}`"
            )
            st.text(item["text"])


def render_answer(result: dict[str, Any]) -> None:
    """Render the answer, citations, warnings, and conflicts."""
    if result["abstained"]:
        st.error(f"**Abstained** — reason: `{result['abstention_reason']}`")
    st.markdown(result["answer"])

    if result["citations"]:
        st.markdown("**Citations**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "chunk_id": c["chunk_id"],
                        "source": c["title"],
                        "version": c["version"],
                        "effective": c["effective_date"],
                        "authority": f"{c['authority_score']}/5",
                        "access": c["acl_status"],
                    }
                    for c in result["citations"]
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    for conflict in result["conflicts"]:
        st.warning(
            f"**Conflicting sources ({conflict['kind']}) — "
            f"{conflict['policy_family']}**\n\n{conflict['description']}"
            + (
                f"\n\n*Preferred for a current-state answer:* "
                f"{conflict['preference_reason']}"
                if conflict.get("preference_reason")
                else ""
            )
        )

    for warning in result["freshness_warnings"]:
        st.info(f"**Freshness:** {warning}")

    sufficiency = result["sufficiency"]
    with st.expander(
        f"Evidence sufficiency: {'PASSED' if sufficiency['sufficient'] else 'FAILED'}"
    ):
        st.write(sufficiency["message"])
        st.dataframe(
            pd.DataFrame(
                [{"check": k, "passed": v} for k, v in sufficiency["checks"].items()]
            ),
            width="stretch",
            hide_index=True,
        )
        st.json(sufficiency["details"])


def render_subquery_trace(trace: dict[str, Any]) -> None:
    """Render the R4 decomposition trace."""
    if trace["route"] != "R4" and len(trace.get("subqueries", [])) <= 1:
        return
    st.subheader("Iterative retrieval trace (R4)")
    st.caption(
        f"{trace['iterations']} iteration(s), capped by "
        "`retrieval.r4_max_iterations` in config/router.yaml."
    )
    for index, subquery in enumerate(trace.get("subqueries", []), start=1):
        st.markdown(f"**{index}.** `{subquery}`")
    if trace.get("per_iteration"):
        st.dataframe(pd.DataFrame(trace["per_iteration"]), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_ask(base: str | None, engine: AHRAGEngine | None, user_id: str) -> None:
    """The main question-answering page."""
    st.header("Ask a question")

    with st.expander("Demo scenarios — click to load one", expanded=False):
        st.caption(
            "Each scenario exercises a different part of the router. Select a "
            "matching user in the sidebar for the intended behaviour."
        )
        scenarios = [
            ("R1 · exact error code", "alice.employee", "What is the remediation for ERR-5041?"),
            ("R2/R3 · conceptual policy", "alice.employee", "Why does the company use a hybrid working model and what is expected of staff?"),
            ("R4 · multi-document comparison", "carol.hr", "Compare the annual leave carry-over rules between the current policy and the 2023 edition"),
            ("R0 · restricted finance, no leakage", "alice.employee", "What is the Q3 2026 revenue forecast and gross margin?"),
            ("Conflict disclosure + freshness", "carol.hr", "What is the current annual leave entitlement now?"),
            ("Abstention · unsupported question", "erin.contractor", "What is the airspeed velocity of an unladen swallow?"),
        ]
        for label, scenario_user, scenario_query in scenarios:
            cols = st.columns([3, 2, 1])
            cols[0].markdown(f"**{label}**")
            cols[1].code(scenario_user, language=None)
            if cols[2].button("Load", key=f"load-{label}"):
                st.session_state["query"] = scenario_query
                st.session_state["pending_user"] = scenario_user
                st.rerun()

    query = st.text_area(
        "Question",
        key="query",
        height=90,
        placeholder="e.g. What is the remediation for ERR-5041?",
    )
    use_history = st.checkbox(
        "Send previous turns as conversation context (enables follow-up handling)",
        value=False,
    )

    if st.button("Ask", type="primary", disabled=not query.strip()):
        history = st.session_state.get("history", []) if use_history else []
        with st.spinner("Routing and retrieving…"):
            if base:
                payload = call_api(
                    base,
                    "POST",
                    "/api/query",
                    json={"query": query, "user_id": user_id, "history": history},
                )
                result = payload["result"]
            else:
                assert engine is not None
                result = engine.answer(query, user_id, history).model_dump(mode="json")
        st.session_state["last_result"] = result
        st.session_state.setdefault("history", []).append(query)

    result = st.session_state.get("last_result")
    if not result:
        return

    decision = result["decision"]
    render_route_badge(decision["route"], decision["confidence"])

    cols = st.columns(4)
    cols[0].metric("Actual latency", f"{result['total_latency_s'] * 1000:.1f} ms")
    cols[1].metric("Est. cost", f"${result['estimated_cost_usd']:.6f}")
    cols[2].metric("Evidence chunks", len(result["evidence"]))
    cols[3].metric("Generator", result["generator"].split(":")[0])

    tabs = st.tabs(["Answer", "Why this route?", "Evidence", "Trace"])
    with tabs[0]:
        render_answer(result)
    with tabs[1]:
        render_why_panel(decision)
    with tabs[2]:
        render_evidence(result["evidence"])
    with tabs[3]:
        render_subquery_trace(result["trace"])
        st.markdown("**Retrieval telemetry**")
        st.json(
            {
                "route": result["trace"]["route"],
                "acl_scoped_pool": result["trace"]["acl_scoped_pool"],
                "acl_withheld": result["trace"]["acl_withheld"],
                "sparse_candidates": result["trace"]["sparse_candidates"],
                "dense_candidates": result["trace"]["dense_candidates"],
                "fused_candidates": result["trace"]["fused_candidates"],
                "reranked_candidates": result["trace"]["reranked_candidates"],
                "timings_ms": result["trace"]["timings_ms"],
            }
        )
        if result.get("audit_id"):
            st.caption(f"Audit record: `{result['audit_id']}`")


def page_corpus(
    base: str | None,
    engine: AHRAGEngine | None,
    manifest_path: str | None = None,
) -> None:
    """Corpus inspection and ingestion page."""
    st.header("Corpus & ingestion")

    if base:
        documents = call_api(base, "GET", "/api/documents")
        roles = call_api(base, "GET", "/api/roles")
    else:
        assert engine is not None
        counts: dict[str, int] = {}
        for chunk in engine.db.get_chunks():
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        documents = [
            {
                **meta.model_dump(mode="json"),
                "chunk_count": counts.get(meta.doc_id, 0),
            }
            for meta in engine.db.get_documents()
        ]
        roles = engine.db.get_roles()

    st.subheader("Documents")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "type": d["doc_type"],
                    "owner": d["owner"],
                    "ACL roles": ", ".join(d["acl_roles"]),
                    "version": d["version"],
                    "effective": d["effective_date"],
                    "authority": f"{d['authority_score']}/5",
                    "family": d.get("policy_family") or "—",
                    "supersedes": d.get("supersedes") or "—",
                    "superseded by": d.get("superseded_by") or "—",
                    "chunks": d.get("chunk_count", 0),
                }
                for d in documents
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Upload a document")
    st.caption(
        "Supported: .txt, .md, .pdf, .docx. A document uploaded without ACL "
        "roles is restricted to hr + manager, never made readable by everyone."
    )
    uploaded = st.file_uploader("File", type=["txt", "md", "pdf", "docx"])
    cols = st.columns(3)
    title = cols[0].text_input("Title", value="")
    doc_type = cols[1].selectbox("Type", [t.value for t in DocType], index=len(DocType) - 1)
    owner = cols[2].text_input("Owner", value="unassigned")
    cols = st.columns(3)
    selected_roles = cols[0].multiselect("ACL roles", roles)
    version = cols[1].text_input("Version", value="1.0")
    authority = cols[2].slider("Authority score", 1, 5, 3)
    cols = st.columns(3)
    family = cols[0].text_input("Policy family (for conflict detection)", value="")
    supersedes = cols[1].text_input("Supersedes doc_id", value="")
    effective = cols[2].date_input("Effective date", value=date.today())

    if st.button("Ingest", type="primary", disabled=uploaded is None):
        if base:
            response = call_api(
                base,
                "POST",
                "/api/ingest",
                files={"file": (uploaded.name, uploaded.getvalue())},
                data={
                    "title": title or uploaded.name,
                    "doc_type": doc_type,
                    "owner": owner,
                    "acl_roles": ",".join(selected_roles),
                    "version": version,
                    "authority_score": str(authority),
                    "policy_family": family,
                    "supersedes": supersedes,
                    "effective_date": effective.isoformat(),
                },
            )
        else:
            assert engine is not None
            upload_dir = Path(engine.settings.data_dir) / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            target = upload_dir / uploaded.name
            target.write_bytes(uploaded.getvalue())
            meta, chunks = engine.ingestion.ingest_file(
                target,
                title=title or uploaded.name,
                doc_type=DocType(doc_type),
                owner=owner,
                acl_roles=selected_roles or None,
                effective_date=effective,
                version=version,
                authority_score=authority,
                policy_family=family or None,
                supersedes=supersedes or None,
            )
            engine.refresh_indexes()
            response = {
                "doc_id": meta.doc_id,
                "chunks": len(chunks),
                "acl_roles": meta.acl_roles,
                "warning": None if selected_roles else "Restricted to the default ACL.",
            }
        st.success(
            f"Ingested `{response['doc_id']}` — {response['chunks']} chunks, "
            f"readable by: {', '.join(response['acl_roles'])}"
        )
        if response.get("warning"):
            st.warning(response["warning"])
        get_local_engine.clear()
        st.rerun()

    corpus_name = next(
        (name for name, path in CORPUS_OPTIONS.items() if path == manifest_path),
        "Selected corpus",
    )
    st.subheader("Re-seed the selected corpus")
    st.caption(f"Drops all documents and chunks, then re-ingests: {corpus_name}.")
    if corpus_name in SLOW_CORPORA:
        st.warning(
            "This corpus takes several minutes to ingest and index, and the "
            "browser will appear to hang while it does — Streamlit cannot show "
            "progress during a synchronous seed. For anything more than a "
            "one-off look, seed it from the command line instead:\n\n"
            "```\n"
            "python -c \"from ahrag.eval.harness import build_seeded_engine, "
            "INTEGRATED_MANIFEST; build_seeded_engine(INTEGRATED_MANIFEST)\"\n"
            "```\n\n"
            "That caches the result under `data/corpus-cache/`, after which "
            "loads take about 15 seconds."
        )
    if base and manifest_path is not None:
        st.info(
            "The API backend seeds its own configured corpus and ignores this "
            "selection. Stop the API to have the UI seed in-process, or "
            "restart the API against the corpus you want."
        )
    if st.button("Re-seed"):
        with st.spinner(f"Ingesting {corpus_name}. This can take minutes."):
            if base:
                result = call_api(base, "POST", "/api/seed")
            else:
                assert engine is not None
                result = engine.seed(reset=True, manifest_path=manifest_path).as_dict()
        st.success(f"Seeded {result['documents']} documents, {result['chunks']} chunks.")
        get_local_engine.clear()
        st.rerun()


def page_audit(base: str | None, engine: AHRAGEngine | None) -> None:
    """Audit-log page."""
    st.header("Audit log")
    st.caption(
        "Records store a SHA-256 query hash and document/chunk identifiers. Raw "
        "query text and evidence text are omitted unless AHRAG_VERBOSE_AUDIT=true "
        "is set for local debugging."
    )

    if base:
        payload = call_api(base, "GET", "/api/audit?limit=200")
        records = payload["records"]
        distribution = payload["route_distribution"]
        verbose = payload["verbose_audit_enabled"]
    else:
        assert engine is not None
        records = engine.db.get_audit_records(limit=200)
        distribution = engine.audit.route_distribution(limit=1000)
        verbose = engine.settings.verbose_audit

    if verbose:
        st.warning(
            "AHRAG_VERBOSE_AUDIT is enabled: raw query text is being written to "
            "the audit log. This is a local debugging mode."
        )

    if not records:
        st.info("No audit records yet. Ask a question on the Ask page.")
        return

    st.subheader("Route distribution")
    st.bar_chart(pd.DataFrame([distribution]).T.rename(columns={0: "queries"}))

    st.subheader("Records")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "timestamp": r["timestamp"],
                    "user": r["user_id"],
                    "roles": ", ".join(r["user_roles"]),
                    "route": r["route"],
                    "conf": round(r["router_confidence"], 2),
                    "query_hash": r["query_hash"][:16] + "…",
                    "docs used": len(r["documents_used"]),
                    "ACL pool": r["acl_pool_size"],
                    "ACL withheld": r["acl_withheld_count"],
                    "citations": len(r["citations"]),
                    "abstained": r["abstained"],
                    "reason": r["abstention_reason"],
                    "conflicts": r["conflicts_disclosed"],
                    "est. $": round(r["estimated_cost_usd"], 6),
                    "latency s": round(r["total_latency_s"], 4),
                }
                for r in records
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    selected = st.selectbox(
        "Inspect a record",
        options=[r["audit_id"] for r in records],
        format_func=lambda a: f"{a[:12]} — {next(r['route'] for r in records if r['audit_id'] == a)}",
    )
    record = next(r for r in records if r["audit_id"] == selected)
    st.json(record)


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a metric for a table cell, distinguishing zero from missing."""
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _stars(p_value: Any) -> str:
    """Conventional significance marker for a p-value."""
    if not isinstance(p_value, (int, float)):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def render_report_baselines(data: dict[str, Any]) -> None:
    """System comparison: per-system metrics, pairwise stats, efficiency."""
    per_system = data.get("per_system") or {}
    systems = data.get("systems") or {}
    if per_system:
        st.markdown("**Per-system metrics** — mean with 95% bootstrap CI.")
        rows = []
        for key, metrics in per_system.items():
            row: dict[str, Any] = {"system": key, "name": systems.get(key, {}).get("name", "")}
            for metric, stats in metrics.items():
                row[metric] = (
                    f"{stats['mean']:.3f} "
                    f"[{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]"
                )
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    comparisons = data.get("comparisons") or {}
    if comparisons:
        proposed = data.get("proposed", "P1")
        st.markdown(
            f"**Pairwise comparisons versus {proposed}** — paired bootstrap "
            "p-value and Cohen's *d*. `***` p<0.001, `**` p<0.01, `*` p<0.05."
        )
        rows = []
        for other, metrics in comparisons.items():
            for metric, stats in metrics.items():
                rows.append({
                    "vs": other,
                    "metric": metric,
                    proposed: _fmt(stats.get("proposed_mean"), 3),
                    "other": _fmt(stats.get("other_mean"), 3),
                    "delta": _fmt(stats.get("delta")),
                    "p": _fmt(stats.get("p_value")),
                    "d": _fmt(stats.get("cohens_d"), 3),
                    "": _stars(stats.get("p_value")),
                    "effect": stats.get("effect_size", ""),
                })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    efficiency = data.get("efficiency") or {}
    if efficiency:
        st.markdown("**Routing and efficiency**")
        rows = []
        for key, info in efficiency.items():
            routes = info.get("route_distribution") or {}
            rows.append({
                "system": key,
                "routes": " ".join(f"{r}:{n}" for r, n in sorted(routes.items())),
                "mean latency s": _fmt(info.get("mean_latency_s")),
                "est. $/query": _fmt(info.get("mean_cost_usd"), 6),
                "abstention rate": _fmt(info.get("abstention_rate"), 3),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    by_type = data.get("recall_by_query_type") or {}
    if by_type:
        st.markdown(
            "**Recall@5 by query type.** Strata with no gold chunks "
            "(`permission_boundary`, `unanswerable`) show no recall by "
            "construction — they are scored by abstention appropriateness."
        )
        rows = []
        for query_type, per_key in sorted(by_type.items()):
            row: dict[str, Any] = {"query type": query_type}
            row.update({k: _fmt(v, 3) for k, v in sorted(per_key.items())})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_report_ablations(data: dict[str, Any]) -> None:
    """Component ablations against the full system."""
    full = data.get("full_system") or {}
    if full:
        st.markdown("**Full system** (the reference row)")
        st.dataframe(
            pd.DataFrame([{k: _fmt(v.get("mean"), 4) for k, v in full.items()}]),
            width="stretch",
            hide_index=True,
        )

    ablations = data.get("ablations") or {}
    if not ablations:
        return
    st.markdown(
        "**Effect of disabling each mechanism.** A negative delta means the "
        "mechanism was helping; a *positive* one means the system is better "
        "without it."
    )
    rows = []
    for key, info in ablations.items():
        for metric, stats in (info.get("metrics") or {}).items():
            rows.append({
                "ablation": key,
                "label": info.get("label", ""),
                "metric": metric,
                "ablated": _fmt(stats.get("ablated_mean"), 3),
                "full": _fmt(stats.get("full_mean"), 3),
                "delta": _fmt(stats.get("delta_vs_full")),
                "p": _fmt(stats.get("p_value")),
                "d": _fmt(stats.get("cohens_d"), 3),
                "": _stars(stats.get("p_value")),
            })
    frame = pd.DataFrame(rows)
    significant = st.checkbox(
        "Show only statistically significant effects (p < 0.05)", value=True
    )
    if significant and not frame.empty:
        frame = frame[frame[""].isin({"*", "**", "***"})]
    st.dataframe(frame, width="stretch", hide_index=True)
    if significant:
        st.caption(
            "Rows hidden by this filter are mechanisms with no measurable "
            "effect on the metric shown — which is itself a result worth "
            "knowing. Untick to see them."
        )


def render_report_embeddings(data: dict[str, Any]) -> None:
    """Embedding backend comparison."""
    backends = data.get("backends") or {}
    if backends:
        rows = []
        for key, info in backends.items():
            row: dict[str, Any] = {
                "backend": key,
                "model": info.get("embedder", ""),
                "dim": info.get("dim"),
            }
            for metric, stats in (info.get("metrics") or {}).items():
                row[metric] = f"{stats['mean']:.3f}"
            agreement = info.get("agreement") or {}
            row["sparse/dense Jaccard"] = _fmt(
                agreement.get("mean_sparse_dense_jaccard"), 3
            )
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Lower sparse/dense Jaccard means dense retrieval is returning "
            "something sparse does not, which is what makes the R1-vs-R2 route "
            "choice consequential."
        )

    comparisons = data.get("comparisons_vs_baseline") or {}
    if comparisons:
        st.markdown(f"**Versus `{data.get('baseline', 'baseline')}`**")
        rows = []
        for key, metrics in comparisons.items():
            for metric, stats in metrics.items():
                rows.append({
                    "backend": key,
                    "metric": metric,
                    "delta": _fmt(stats.get("delta")),
                    "p": _fmt(stats.get("p_value")),
                    "d": _fmt(stats.get("cohens_d"), 3),
                    "": _stars(stats.get("p_value")),
                    "effect": stats.get("effect_size", ""),
                })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_report_specialisation(data: dict[str, Any]) -> None:
    """Scope-pure index specialisation: lattice, interference, quality."""
    lattice = data.get("lattice") or {}
    if lattice:
        st.markdown("**ACL lattice** — the structure specialisation works over.")
        columns = st.columns(4)
        columns[0].metric("Distinct ACL classes", lattice.get("distinct_classes", "—"))
        columns[1].metric("Principals", lattice.get("principals", "—"))
        columns[2].metric("Total chunks", lattice.get("total_chunks", "—"))
        columns[3].metric(
            "Smallest class share",
            _fmt(lattice.get("smallest_class_fraction"), 3),
        )
        if lattice.get("specialisation_is_degenerate"):
            st.error(
                "Every principal has the same authorised scope on this corpus, "
                "so specialisation is provably a no-op here and the results "
                "below are vacuous."
            )
        st.caption(f"Class sizes (chunks): {lattice.get('class_sizes')}")

    interference = data.get("interference") or {}
    if interference:
        st.markdown(
            "**Interference from unreadable documents.** A principal's "
            "authorised subcorpus is held fixed while everything outside it is "
            "deleted. A non-interfering system must return identical results: "
            "tau = 1.000 with zero changes."
        )
        rows = []
        for label, info in interference.items():
            rows.append({
                "condition": label,
                "comparisons": info.get("comparisons"),
                "Kendall tau": _fmt(info.get("mean_kendall_tau"), 3),
                "top-1 flips": _fmt(info.get("top1_flip_rate"), 3),
                "order changes": _fmt(info.get("order_change_rate"), 3),
                "route changes": _fmt(info.get("route_change_rate"), 3),
                "non-interfering": "yes" if info.get("non_interfering") else "NO",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    quality = data.get("quality") or {}
    frontier = quality.get("frontier") or {}
    if frontier:
        st.markdown(
            "**Purity / utility frontier.** `lambda` is an information-flow "
            "budget for the sparse channel: 0 is provably non-interfering, 1 "
            "restores global statistics."
        )
        rows = []
        for label, entry in frontier.items():
            row: dict[str, Any] = {
                "setting": label,
                "provably pure": "yes" if entry.get("provably_pure") else "no",
            }
            for metric in ("recall_at_5", "mrr", "ndcg_at_10",
                           "abstention_appropriateness"):
                stats = entry.get(metric) or {}
                if stats:
                    row[metric] = (
                        f"{stats.get('mean', float('nan')):.3f} "
                        f"({stats.get('delta', 0):+.4f}) "
                        f"{_stars(stats.get('p_value'))}"
                    )
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Delta is against the unspecialised system. Purity is expected to "
            "cost a little quality; the point of the curve is that the cost is "
            "measured rather than assumed."
        )


REPORT_RENDERERS = {
    "baselines": render_report_baselines,
    "baselines_heldout": render_report_baselines,
    "ablations": render_report_ablations,
    "embeddings": render_report_embeddings,
    "specialisation": render_report_specialisation,
}


def page_evaluation(base: str | None, engine: AHRAGEngine | None) -> None:
    """Evaluation page: experiment reports, then stored in-database runs."""
    st.header("Evaluation")

    reports = load_reports()

    st.markdown(
        "Experiments are run from a terminal, never from the UI, so every "
        "number shown here comes from an explicit, reproducible command. This "
        "page renders whatever reports exist in `data/reports/`.\n\n"
        "```bash\n"
        "python -m ahrag.evaluate --per-type\n"
        "python improvement_files/baseline_code/compare_baselines.py --integrated --by-type\n"
        "python improvement_files/baseline_code/ablation_study.py --integrated\n"
        "python improvement_files/evaluation_tools/compare_embeddings.py --integrated\n"
        "python improvement_files/evaluation_tools/measure_specialisation.py --integrated\n"
        "```"
    )

    with st.expander("The eight systems compared", expanded=False):
        st.markdown(
            "- **B1** fixed sparse BM25 (always R1)\n"
            "- **B2** fixed dense vector (always R2)\n"
            "- **B3** fixed hybrid RRF (always R3)\n"
            "- **B4** always-maximal decomposed iterative (always R4)\n"
            "- **B5** complexity-only router — hand-set token thresholds\n"
            "- **B6** Adaptive-RAG — a *trained* classifier over query-text "
            "features only, which is the shape of Jeong et al. (2024)\n"
            "- **P1** AHRAG governance-aware router — hard constraints, then "
            "hand-tuned utility over the admissible routes\n"
            "- **P2** AHRAG learned router — the same hard constraints, with a "
            "trained classifier ranking the admissible routes\n\n"
            "All eight share one corpus, chunking, indexes, reranker, evidence "
            "gates, generator and ACL layer. Only the routing policy differs, "
            "which is what makes the comparison an ablation of routing rather "
            "than of unrelated pipelines."
        )

    if not reports:
        st.info(
            "No experiment reports found in `data/reports/` yet. Run one of the "
            "commands above; each writes its report there automatically."
        )
    else:
        st.subheader("Experiment reports")
        labels = [
            f"{report.title}  ·  {report.corpus_label}" for report in reports
        ]
        chosen = st.selectbox("Report", labels, index=0)
        report = reports[labels.index(chosen)]

        columns = st.columns(3)
        columns[0].caption(f"**Corpus:** {report.corpus_label}")
        columns[1].caption(f"**Generated:** {report.generated_at}")
        columns[2].caption(f"**Items:** {report.data.get('items', '—')}")
        if report.data.get("embedder"):
            st.caption(f"**Embedding backend:** {report.data['embedder']}")

        renderer = REPORT_RENDERERS.get(report.kind)
        if renderer is None:
            st.warning(
                f"No renderer for report kind `{report.kind}`; showing raw JSON."
            )
            st.json(report.data)
        else:
            renderer(report.data)

        with st.expander("Raw report JSON", expanded=False):
            st.json(report.data)

    st.divider()
    st.subheader("Stored in-database runs")
    st.caption(
        "Written by `python -m ahrag.evaluate`. These are separate from the "
        "file-based experiment reports above."
    )

    if base:
        runs = call_api(base, "GET", "/api/eval-runs")["runs"]
    else:
        assert engine is not None
        runs = engine.db.get_eval_runs(limit=20)

    if not runs:
        st.info(
            "No evaluation runs recorded in this database. Note that "
            "`python -m ahrag.evaluate` writes to a separate evaluation database "
            "(`data/ahrag_eval.sqlite3`) by default; pass `--db data/ahrag.sqlite3` "
            "to record runs here instead."
        )
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "run_id": r["run_id"],
                    "started": r["started_at"],
                    "finished": r["finished_at"] or "—",
                    "config": r["config_hash"],
                    "notes": r["notes"],
                }
                for r in runs
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    latest = runs[0]
    summary = latest.get("summary") or {}
    summaries = summary.get("summaries", {})
    if not summaries:
        st.warning("The most recent run has no stored summary.")
        return

    st.markdown(f"**Latest run — {latest['run_id']}**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "system": key,
                    "Recall@5": s.get("recall_at_5"),
                    "MRR": s.get("mrr"),
                    "nDCG@10": s.get("ndcg_at_10"),
                    "Cit. precision": s.get("citation_precision"),
                    "Cit. coverage": s.get("citation_coverage"),
                    "Abstention ✓": s.get("abstention_appropriateness"),
                    "ACL violations": s.get("acl_violations"),
                    "Freshness": s.get("freshness_compliance"),
                    "Mean latency s": s.get("mean_latency_s"),
                    "p95 latency s": s.get("p95_latency_s"),
                    "Est. $/query": s.get("mean_cost_usd"),
                }
                for key, s in summaries.items()
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**Route distribution by system**")
    st.dataframe(
        pd.DataFrame(
            [{"system": key, **s.get("route_distribution", {})} for key, s in summaries.items()]
        ),
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(
        page_title="AHRAG — Adaptive Hybrid RAG", page_icon="🧭", layout="wide"
    )
    st.title("🧭 AHRAG — Adaptive Hybrid Retrieval-Augmented Generation")
    st.caption(
        "Governance-aware adaptive routing for enterprise RAG. "
        "Research prototype — see RESEARCH_LIMITATIONS.md."
    )

    base = api_base()
    engine: AHRAGEngine | None = None

    with st.sidebar:
        st.subheader("Session")
        if base:
            corpus_name = st.selectbox(
                "Corpus",
                list(CORPUS_OPTIONS),
                index=0,
                disabled=True,
                help="Corpus selection is controlled by the FastAPI backend.",
            )
            manifest_path = None
            st.success(f"Connected to API at {base}")
            st.caption("API mode uses the server's configured demo corpus.")
            users = call_api(base, "GET", "/api/users")
            backends = call_api(base, "GET", "/api/health")["backends"]
        else:
            corpus_name = st.selectbox("Corpus", list(CORPUS_OPTIONS))
            manifest_path = CORPUS_OPTIONS[corpus_name]
            engine = get_local_engine(manifest_path)
            st.info("Running in-process (API backend not reachable).")
            assert engine is not None
            st.caption(f"Active corpus: {corpus_name}")
            users = [
                {
                    "user_id": u.user_id,
                    "display_name": u.display_name,
                    "roles": u.roles,
                    "description": u.description,
                    "authorised_chunks": len(engine.acl.scope_for(u).allowed_chunk_ids),
                    "total_chunks": engine.acl.total_chunks,
                    "withheld_chunks": engine.acl.scope_for(u).withheld_count,
                }
                for u in engine.db.get_users()
            ]
            backends = engine.backend_info()

        if not users:
            st.error("No demo users found. Re-seed the corpus on the Corpus page.")
            st.stop()

        ids = [u["user_id"] for u in users]
        default_index = 0
        pending = st.session_state.pop("pending_user", None)
        if pending in ids:
            default_index = ids.index(pending)
        elif st.session_state.get("user_id") in ids:
            default_index = ids.index(st.session_state["user_id"])

        user_id = st.selectbox(
            "Signed in as",
            ids,
            index=default_index,
            format_func=lambda uid: next(
                u["display_name"] for u in users if u["user_id"] == uid
            ),
        )
        st.session_state["user_id"] = user_id
        selected = next(u for u in users if u["user_id"] == user_id)

        st.markdown(f"**Roles:** `{'`, `'.join(selected['roles'])}`")
        st.caption(selected["description"])
        st.metric(
            "Authorised scope",
            f"{selected['authorised_chunks']} / {selected['total_chunks']} chunks",
            delta=f"-{selected['withheld_chunks']} withheld",
            delta_color="off",
        )
        st.caption(
            "ACL is applied **before** routing and retrieval. Withheld chunks "
            "are never candidates, so they cannot influence the route, the "
            "ranking, the answer, or the log."
        )

        st.divider()
        st.subheader("Active backends")
        for key, value in backends.items():
            st.caption(f"**{key}**: `{value}`")

        st.divider()
        if st.button("Clear conversation history"):
            st.session_state["history"] = []
            st.session_state.pop("last_result", None)
            st.rerun()
        st.caption(f"AHRAG v{__version__}")

    page = st.radio(
        "Page",
        ["Ask", "Corpus & ingestion", "Audit log", "Evaluation"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Ask":
        page_ask(base, engine, user_id)
    elif page == "Corpus & ingestion":
        page_corpus(base, engine, manifest_path)
    elif page == "Audit log":
        page_audit(base, engine)
    else:
        page_evaluation(base, engine)


main()
