"""Initial schema — users, datasets, transactions, analysis_results

Revision ID: 001_initial
Revises: 
Create Date: 2025-02-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("clerk_id", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("email", sa.String(320), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Datasets
    op.create_table(
        "datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("row_count", sa.Integer, default=0),
        sa.Column("status", sa.String(32), default="uploaded"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Transactions
    op.create_table(
        "transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "dataset_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("datasets.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("transaction_id", sa.String(128), nullable=True),
        sa.Column("sender_id", sa.String(256), nullable=False),
        sa.Column("receiver_id", sa.String(256), nullable=False),
        sa.Column("amount", sa.Float, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_transactions_dataset_sender", "transactions", ["dataset_id", "sender_id"])
    op.create_index("ix_transactions_dataset_receiver", "transactions", ["dataset_id", "receiver_id"])

    # Analysis Results
    op.create_table(
        "analysis_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "dataset_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("datasets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("celery_task_id", sa.String(255), nullable=True, index=True),
        sa.Column("status", sa.String(32), default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("graph_json", sa.Text, nullable=True),
        sa.Column("risk_json", sa.Text, nullable=True),
        sa.Column("flags_json", sa.Text, nullable=True),
        sa.Column("stats_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("analysis_results")
    op.drop_table("transactions")
    op.drop_table("datasets")
    op.drop_table("users")
