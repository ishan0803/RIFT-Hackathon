"""SQLAlchemy ORM models for the AML Network Analyzer."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


# ── User (synced from Clerk) ─────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clerk_id = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(320), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    datasets = relationship("Dataset", back_populates="user", cascade="all, delete-orphan")
    analyses = relationship("AnalysisResult", back_populates="user", cascade="all, delete-orphan")


# ── Dataset ───────────────────────────────────────────────────
class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(512), nullable=False)
    row_count = Column(Integer, default=0)
    status = Column(String(32), default="uploaded")  # uploaded | parsed | analyzing | completed | failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="datasets")
    transactions = relationship("Transaction", back_populates="dataset", cascade="all, delete-orphan")
    analyses = relationship("AnalysisResult", back_populates="dataset", cascade="all, delete-orphan")


# ── Transaction (raw rows from ingested CSV) ──────────────────
class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    transaction_id = Column(String(128), nullable=True)  # original ID from the CSV
    sender_id = Column(String(256), nullable=False)
    receiver_id = Column(String(256), nullable=False)
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("Dataset", back_populates="transactions")

    __table_args__ = (
        Index("ix_transactions_dataset_sender", "dataset_id", "sender_id"),
        Index("ix_transactions_dataset_receiver", "dataset_id", "receiver_id"),
    )


# ── Analysis Result ───────────────────────────────────────────
class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    celery_task_id = Column(String(255), nullable=True, index=True)
    status = Column(String(32), default="pending")  # pending | running | completed | failed
    error_message = Column(Text, nullable=True)

    # JSON payloads stored as text (parsed on read)
    graph_json = Column(Text, nullable=True)   # {nodes: [...], links: [...]}
    risk_json = Column(Text, nullable=True)    # [{account_id, score, risk_level, reasons}, ...]
    flags_json = Column(Text, nullable=True)   # {cycles: [...], fan_in: [...], ...}
    stats_json = Column(Text, nullable=True)   # {total_nodes, high_risk_count, ...}

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("Dataset", back_populates="analyses")
    user = relationship("User", back_populates="analyses")
