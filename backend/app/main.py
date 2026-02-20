"""
FastAPI application entry point — AML Network Analyzer backend.

Mounts all routers, configures CORS, structured logging, and
global exception handling.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import analysis, network, tasks, upload
from app.core.config import settings

# ── Structured Logging ────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# ── Application ───────────────────────────────────────────────
app = FastAPI(
    title="AML Network Analyzer API",
    description="Production-grade API for Anti-Money Laundering network analysis, "
                "risk scoring, and subgraph pattern hunting.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ── Global Exception Handler ─────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


# ── Mount Routers ─────────────────────────────────────────────
app.include_router(upload.router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(network.router, prefix="/api/v1")
app.include_router(tasks.router, prefix="/api/v1")


# ── Health Check ──────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "AML Network Analyzer"}


@app.on_event("startup")
async def startup_event():
    logger.info("app_started", cors_origins=settings.cors_origin_list)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("app_shutdown")
