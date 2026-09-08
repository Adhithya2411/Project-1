"""FastAPI application.

The engine is constructed once at startup and stored on ``app.state``, not in a
module global, so tests can build an app over a temporary database without
polluting anything. Routes are thin: all logic lives in the pipeline.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile

from .. import __version__
from ..config import RouterConfig, Settings, get_settings
from ..db import Database
from ..ingestion.loaders import LoaderError, supported_extensions
from ..ingestion.pipeline import DEFAULT_UPLOAD_ACL
from ..models import DocType
from ..pipeline import AHRAGEngine, UnknownUserError
from .schemas import (
    AuditResponse,
    DocumentSummary,
    EvalRunsResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SeedResponse,
    UserSummary,
)

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def get_engine(request: Request) -> AHRAGEngine:
    """FastAPI dependency returning the app's engine."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:  # pragma: no cover - startup guarantees this
        raise HTTPException(status_code=503, detail="Engine is not initialised.")
    return engine


def create_app(
    settings: Settings | None = None,
    engine: AHRAGEngine | None = None,
    seed_on_startup: bool = True,
    today: date | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Operational settings. Defaults to process settings.
        engine: A pre-built engine. Supplied by tests; otherwise one is created.
        seed_on_startup: Seed the demo corpus when it is empty.
        today: Fixed reference date for freshness comparisons.

    Returns:
        A configured :class:`FastAPI` application.
    """
    resolved = settings or get_settings()
    app = FastAPI(
        title="AHRAG — Adaptive Hybrid RAG",
        version=__version__,
        description=(
            "Local research prototype of a governance-aware adaptive router for "
            "enterprise RAG. Not a production system; see RESEARCH_LIMITATIONS.md."
        ),
    )

    if engine is None:
        config = RouterConfig.load(resolved.router_config)
        db = Database(resolved.db_path)
        engine = AHRAGEngine(settings=resolved, config=config, db=db, today=today)
        if seed_on_startup:
            try:
                engine.ensure_seeded()
            except (FileNotFoundError, ValueError) as exc:
                logger.error("Could not seed demo corpus at startup: %s", exc)

    app.state.engine = engine
    app.state.settings = resolved

    # -- health & metadata -------------------------------------------------

    @app.get("/api/health", response_model=HealthResponse)
    def health(engine: AHRAGEngine = Depends(get_engine)) -> HealthResponse:
        """Report status and which retrieval/generation backends are active."""
        return HealthResponse(
            status="ok", version=__version__, backends=engine.backend_info()
        )

    @app.get("/api/users", response_model=list[UserSummary])
    def list_users(engine: AHRAGEngine = Depends(get_engine)) -> list[UserSummary]:
        """List demo users with the size of each one's authorised scope."""
        out: list[UserSummary] = []
        for user in engine.db.get_users():
            scope = engine.acl.scope_for(user)
            out.append(
                UserSummary(
                    user_id=user.user_id,
                    display_name=user.display_name,
                    roles=user.roles,
                    description=user.description,
                    authorised_chunks=len(scope.allowed_chunk_ids),
                    total_chunks=scope.total_chunks,
                    withheld_chunks=scope.withheld_count,
                )
            )
        return out

    @app.get("/api/documents", response_model=list[DocumentSummary])
    def list_documents(engine: AHRAGEngine = Depends(get_engine)) -> list[DocumentSummary]:
        """List corpus documents and their governance metadata."""
        counts: dict[str, int] = {}
        for chunk in engine.db.get_chunks():
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        return [
            DocumentSummary(
                doc_id=meta.doc_id,
                title=meta.title,
                doc_type=meta.doc_type,
                owner=meta.owner,
                acl_roles=meta.acl_roles,
                version=meta.version,
                effective_date=meta.effective_date.isoformat(),
                created_date=meta.created_date.isoformat(),
                authority_score=meta.authority_score,
                policy_family=meta.policy_family,
                supersedes=meta.supersedes,
                superseded_by=meta.superseded_by,
                chunk_count=counts.get(meta.doc_id, 0),
                source_uri=meta.source_uri,
            )
            for meta in engine.db.get_documents()
        ]

    @app.get("/api/roles", response_model=list[str])
    def list_roles(engine: AHRAGEngine = Depends(get_engine)) -> list[str]:
        """List every known role."""
        return engine.db.get_roles()

    # -- query -------------------------------------------------------------

    @app.post("/api/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest, engine: AHRAGEngine = Depends(get_engine)
    ) -> QueryResponse:
        """Answer a question as the given demo user."""
        try:
            result = engine.answer(payload.query, payload.user_id, payload.history)
        except UnknownUserError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except PermissionError as exc:
            # An ACL invariant breach. Fail loudly rather than degrading: this
            # is a defect, and returning a partial answer would hide it.
            logger.error("Access-control invariant breached: %s", exc)
            raise HTTPException(
                status_code=500, detail="Access-control invariant breach; see server log."
            ) from exc
        return QueryResponse(result=result)

    # -- corpus ------------------------------------------------------------

    @app.post("/api/seed", response_model=SeedResponse)
    def seed(engine: AHRAGEngine = Depends(get_engine)) -> SeedResponse:
        """Re-seed the demo corpus from the packaged manifest."""
        try:
            result = engine.seed(reset=True)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return SeedResponse(**result.as_dict())

    @app.post("/api/ingest", response_model=IngestResponse)
    async def ingest(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        doc_type: str = Form(default="other"),
        owner: str = Form(default="unassigned"),
        acl_roles: str = Form(default=""),
        version: str = Form(default="1.0"),
        authority_score: int = Form(default=3),
        policy_family: str | None = Form(default=None),
        supersedes: str | None = Form(default=None),
        effective_date: str | None = Form(default=None),
        engine: AHRAGEngine = Depends(get_engine),
    ) -> IngestResponse:
        """Ingest an uploaded document into the corpus.

        A document uploaded without ACL roles is stored with a restrictive
        default, never with universal access, and the response says so.
        """
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in supported_extensions():
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type {suffix!r}. Supported: "
                    f"{sorted(supported_extensions())}"
                ),
            )

        body = await file.read()
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
            )

        roles = [r.strip().lower() for r in acl_roles.split(",") if r.strip()]
        warning = None
        if not roles:
            warning = (
                "No ACL roles were supplied, so this document was restricted to "
                f"{', '.join(DEFAULT_UPLOAD_ACL)} rather than made readable by "
                "everyone."
            )

        upload_dir = Path(engine.settings.data_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            delete=False, dir=upload_dir, suffix=suffix
        ) as handle:
            handle.write(body)
            temp_path = Path(handle.name)

        try:
            parsed_date = date.fromisoformat(effective_date) if effective_date else None
        except ValueError as exc:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422, detail=f"Invalid effective_date: {exc}"
            ) from exc

        try:
            meta, chunks = engine.ingestion.ingest_file(
                temp_path,
                title=title or Path(file.filename or temp_path.name).stem,
                doc_type=DocType(doc_type),
                owner=owner,
                acl_roles=roles or None,
                effective_date=parsed_date,
                version=version,
                authority_score=authority_score,
                policy_family=policy_family or None,
                supersedes=supersedes or None,
            )
        except LoaderError as exc:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Keep the uploaded file: source_uri points at it, and the evidence
        # table renders that path as the document's provenance.
        stable_path = upload_dir / f"{meta.doc_id}{suffix}"
        shutil.move(str(temp_path), stable_path)

        engine.refresh_indexes()
        return IngestResponse(
            doc_id=meta.doc_id,
            title=meta.title,
            chunks=len(chunks),
            acl_roles=meta.acl_roles,
            warning=warning,
        )

    # -- audit & evaluation ------------------------------------------------

    @app.get("/api/audit", response_model=AuditResponse)
    def audit(
        limit: int = 100, engine: AHRAGEngine = Depends(get_engine)
    ) -> AuditResponse:
        """Return recent audit records and the observed route distribution."""
        limit = max(1, min(limit, 1000))
        return AuditResponse(
            records=engine.db.get_audit_records(limit=limit),
            route_distribution=engine.audit.route_distribution(limit=1000),
            verbose_audit_enabled=engine.settings.verbose_audit,
        )

    @app.delete("/api/audit")
    def clear_audit(engine: AHRAGEngine = Depends(get_engine)) -> dict[str, str]:
        """Clear the audit log (demo convenience)."""
        engine.db.clear_audit()
        return {"status": "cleared"}

    @app.get("/api/eval-runs", response_model=EvalRunsResponse)
    def eval_runs(engine: AHRAGEngine = Depends(get_engine)) -> EvalRunsResponse:
        """Return stored evaluation runs, newest first."""
        return EvalRunsResponse(runs=engine.db.get_eval_runs(limit=20))

    return app


app = None  # populated by `ahrag.api.main` / uvicorn factory


def main() -> None:  # pragma: no cover - console entry point
    """Run the API with uvicorn using the configured host and port."""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    uvicorn.run(
        "ahrag.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
