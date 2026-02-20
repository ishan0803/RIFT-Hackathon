"""Upload router — ingest CSV/JSON files, parse into PostgreSQL, delete file."""

from __future__ import annotations

import io
import uuid
from datetime import datetime

import pandas as pd
import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.database import get_db
from app.models.models import Dataset, Transaction, User
from app.schemas.schemas import DatasetUploadResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])


async def _get_or_create_user(session: AsyncSession, clerk_id: str) -> User:
    """Find existing user by Clerk ID or create a new record."""
    result = await session.execute(select(User).where(User.clerk_id == clerk_id))
    user = result.scalar_one_or_none()
    if user:
        return user

    user = User(id=uuid.uuid4(), clerk_id=clerk_id)
    session.add(user)
    await session.flush()
    return user


@router.post(
    "",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a transaction file",
)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a CSV or JSON transaction file, parse it, store every row in the
    ``transactions`` table, and delete the raw file. No file is kept on disk.

    Expected columns: ``transaction_id``, ``sender_id``, ``receiver_id``,
    ``amount``, ``timestamp``.
    """
    # ── Validate file type ────────────────────────────────────
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("csv", "json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV and JSON files are accepted.",
        )

    # ── Read file into memory ─────────────────────────────────
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file."
        )

    try:
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_json(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse file: {exc}",
        ) from exc

    # ── Validate required columns ─────────────────────────────
    required = {"transaction_id", "sender_id", "receiver_id", "amount", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required columns: {', '.join(sorted(missing))}. "
                   f"Expected: transaction_id, sender_id, receiver_id, amount, timestamp. "
                   f"Found: {', '.join(df.columns)}",
        )

    # ── Ensure user exists ────────────────────────────────────
    user = await _get_or_create_user(db, user_id)

    # ── Create dataset record ─────────────────────────────────
    dataset = Dataset(
        id=uuid.uuid4(),
        user_id=user.id,
        filename=filename,
        row_count=len(df),
        status="parsed",
    )
    db.add(dataset)
    await db.flush()

    # ── Ingest rows into transactions table ───────────────────
    transactions = []
    for _, row in df.iterrows():
        ts = None
        if "timestamp" in df.columns and pd.notna(row.get("timestamp")):
            try:
                ts = pd.to_datetime(row["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
            except Exception:
                ts = None

        transactions.append(
            Transaction(
                id=uuid.uuid4(),
                dataset_id=dataset.id,
                transaction_id=str(row.get("transaction_id", "")),
                sender_id=str(row["sender_id"]),
                receiver_id=str(row["receiver_id"]),
                amount=float(row["amount"]),
                timestamp=ts,
            )
        )

    db.add_all(transactions)

    logger.info(
        "file_ingested",
        filename=filename,
        rows=len(transactions),
        dataset_id=str(dataset.id),
        user_clerk_id=user_id,
    )

    return DatasetUploadResponse(
        dataset_id=dataset.id,
        filename=filename,
        row_count=len(transactions),
        message=f"Successfully ingested {len(transactions)} transactions.",
    )
