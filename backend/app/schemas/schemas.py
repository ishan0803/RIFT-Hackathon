"""Pydantic schemas for request / response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════

class DatasetResponse(BaseModel):
    id: UUID
    filename: str
    row_count: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetUploadResponse(BaseModel):
    dataset_id: UUID
    filename: str
    row_count: int
    message: str = "File ingested successfully."


# ═══════════════════════════════════════════════════════════════
#  Analysis
# ═══════════════════════════════════════════════════════════════

class AnalysisStartRequest(BaseModel):
    dataset_id: UUID


class AnalysisStartResponse(BaseModel):
    analysis_id: UUID
    celery_task_id: str
    message: str = "Analysis pipeline started."


class AnalysisStatusResponse(BaseModel):
    analysis_id: UUID
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AnalysisResultResponse(BaseModel):
    analysis_id: UUID
    dataset_id: UUID
    status: str
    graph_data: Optional[Any] = None
    risk_data: Optional[List[Any]] = None
    flags: Optional[Any] = None
    stats: Optional[Any] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class AnalysisHistoryItem(BaseModel):
    id: UUID
    dataset_id: UUID
    filename: str
    status: str
    row_count: int
    stats: Optional[Any] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════
#  Network / Isomorphism
# ═══════════════════════════════════════════════════════════════

class IsomorphismRequest(BaseModel):
    analysis_id: UUID
    target_node: str = Field(..., min_length=1, description="Node ID to find structural clones of")
    hops: int = Field(default=1, ge=1, description="Radius around target node")


class IsomorphismStartResponse(BaseModel):
    celery_task_id: str
    message: str = "Isomorphism search started."


class IsomorphismResultResponse(BaseModel):
    match_nodes: List[str]
    match_edges: List[List[str]]
    match_count: int


# ═══════════════════════════════════════════════════════════════
#  Graph Payload (D3-ready)
# ═══════════════════════════════════════════════════════════════

class GraphNode(BaseModel):
    id: str
    color: str
    radius: float
    is_match: int = 0
    title: str


class GraphLink(BaseModel):
    source: str
    target: str
    is_match: int = 0


class GraphPayload(BaseModel):
    nodes: List[GraphNode]
    links: List[GraphLink]


# ═══════════════════════════════════════════════════════════════
#  Risk
# ═══════════════════════════════════════════════════════════════

class RiskEntry(BaseModel):
    account_id: str
    score: int
    risk_level: str
    reasons: str


# ═══════════════════════════════════════════════════════════════
#  Task Polling
# ═══════════════════════════════════════════════════════════════

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # PENDING | STARTED | SUCCESS | FAILURE | RETRY
    result: Optional[Any] = None
    error: Optional[str] = None
