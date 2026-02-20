"""Task status router — poll Celery task progress."""

from __future__ import annotations

import structlog
from celery.result import AsyncResult
from fastapi import APIRouter, Depends

from app.core.auth import get_current_user
from app.schemas.schemas import TaskStatusResponse
from app.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll the status of a Celery task",
)
async def get_task_status(
    task_id: str,
    _user_id: str = Depends(get_current_user),
):
    """
    Check the status of a background task (analysis pipeline or isomorphism search).
    Returns PENDING, STARTED, SUCCESS, FAILURE, or RETRY.
    """
    result = AsyncResult(task_id, app=celery_app)

    response = TaskStatusResponse(
        task_id=task_id,
        status=result.status,
    )

    if result.successful():
        response.result = result.result
    elif result.failed():
        response.error = str(result.result)

    return response
