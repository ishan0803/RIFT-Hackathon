"""Network router — isomorphism search and graph retrieval."""

from __future__ import annotations

import json
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.database import get_db
from app.models.models import AnalysisResult, User
from app.schemas.schemas import IsomorphismRequest, IsomorphismStartResponse
from app.tasks.analysis_tasks import run_isomorphism_search

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/network", tags=["Network"])


@router.post(
    "/isomorphism",
    response_model=IsomorphismStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a VF2 isomorphism search",
)
async def start_isomorphism(
    body: IsomorphismRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Dispatch a Celery task to find structural clones via VF2 isomorphism.
    No size restrictions on the search.
    """
    # Verify analysis exists and belongs to user
    result = await db.execute(
        select(AnalysisResult)
        .join(User, AnalysisResult.user_id == User.id)
        .where(
            AnalysisResult.id == body.analysis_id,
            AnalysisResult.status == "completed",
            User.clerk_id == user_id,
        )
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Completed analysis not found or access denied.",
        )

    # Dispatch isomorphism task
    task = run_isomorphism_search.delay(
        str(analysis.id),
        str(analysis.dataset_id),
        body.target_node,
        body.hops,
    )

    logger.info(
        "isomorphism_dispatched",
        analysis_id=str(analysis.id),
        target_node=body.target_node,
        hops=body.hops,
        celery_task_id=task.id,
    )

    return IsomorphismStartResponse(
        celery_task_id=task.id,
        message=f"Isomorphism search started for node '{body.target_node}' with {body.hops} hop(s).",
    )


@router.get(
    "/graph/{analysis_id}",
    summary="Get the full graph payload for visualization",
)
async def get_graph(
    analysis_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the full D3-ready graph + risk data for a completed analysis."""
    result = await db.execute(
        select(AnalysisResult)
        .join(User, AnalysisResult.user_id == User.id)
        .where(AnalysisResult.id == analysis_id, User.clerk_id == user_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found or access denied.",
        )

    if analysis.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Analysis is not complete yet. Current status: {analysis.status}",
        )

    return {
        "graph": json.loads(analysis.graph_json) if analysis.graph_json else None,
        "risk": json.loads(analysis.risk_json) if analysis.risk_json else None,
        "stats": json.loads(analysis.stats_json) if analysis.stats_json else None,
    }
