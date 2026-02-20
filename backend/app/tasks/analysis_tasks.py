"""
Celery tasks for long-running graph analysis and isomorphism searches.

These run in a worker process to avoid blocking the FastAPI event loop.
They use synchronous SQLAlchemy (psycopg2) since Celery workers are sync.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import networkx as nx
import pandas as pd
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.services.graph_service import (
    analyze_networks,
    assign_risk_scores,
    build_graph_payload,
    build_structured_output,
    flags_to_json,
    flags_from_json,
)
from app.services.isomorphism_service import find_structural_clones
from app.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)

# ── Synchronous DB engine for Celery workers ──────────────────
_sync_engine = create_engine(settings.DATABASE_URL_SYNC, pool_pre_ping=True)
SyncSession = sessionmaker(bind=_sync_engine)


def _load_transactions_df(session: Session, dataset_id: str) -> pd.DataFrame:
    """Load all transactions for a dataset from PostgreSQL into a DataFrame."""
    result = session.execute(
        text(
            """
            SELECT transaction_id, sender_id, receiver_id, amount, timestamp
            FROM transactions
            WHERE dataset_id = :did
            ORDER BY timestamp
            """
        ),
        {"did": dataset_id},
    )
    rows = result.fetchall()
    if not rows:
        raise ValueError(f"No transactions found for dataset {dataset_id}")

    return pd.DataFrame(
        rows, columns=["transaction_id", "sender_id", "receiver_id", "amount", "timestamp"]
    ).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    )


# ═══════════════════════════════════════════════════════════════
#  Task: Full Analysis Pipeline
# ═══════════════════════════════════════════════════════════════

@celery_app.task(bind=True, name="analysis.run_pipeline")
def run_analysis_pipeline(self, analysis_id: str, dataset_id: str):
    """
    Full analysis pipeline:
    1. Load transactions from PostgreSQL
    2. Build graph and detect patterns
    3. Score risks
    4. Build D3 payload
    5. Save results back to PostgreSQL
    """
    logger.info("analysis_pipeline_start", analysis_id=analysis_id, dataset_id=dataset_id)

    with SyncSession() as session:
        try:
            # Mark as running
            session.execute(
                text("UPDATE analysis_results SET status = 'running' WHERE id = :aid"),
                {"aid": analysis_id},
            )
            session.commit()

            # 1. Load data
            t_start = time.monotonic()
            df = _load_transactions_df(session, dataset_id)
            logger.info("data_loaded", rows=len(df))

            # 2. Analyze
            G, flags = analyze_networks(df)

            # 3. Risk scoring
            risk_df = assign_risk_scores(list(G.nodes()), flags)

            # 4. Build graph payload (for D3 visualization)
            graph_payload = build_graph_payload(df, risk_df)

            # 5. Build structured output (user-facing JSON)
            processing_time = time.monotonic() - t_start
            structured_output = build_structured_output(
                G, flags, risk_df, processing_time
            )

            # 6. Compute internal stats (for dashboard cards)
            high_risk_count = int((risk_df["score"] >= 40).sum())
            medium_risk_count = int(((risk_df["score"] > 0) & (risk_df["score"] < 40)).sum())
            stats = {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "total_transactions": len(df),
                "high_risk_count": high_risk_count,
                "medium_risk_count": medium_risk_count,
                "cycles_detected": len(flags["cycles"]),
                "fan_in_detected": len(flags["fan_in"]),
                "fan_out_detected": len(flags["fan_out"]),
                "shells_detected": len(flags["shells"]),
                **structured_output,
            }

            # 7. Persist results
            risk_records = risk_df.to_dict("records")

            session.execute(
                text(
                    """
                    UPDATE analysis_results
                    SET status = 'completed',
                        graph_json = :graph,
                        risk_json = :risk,
                        flags_json = :flags,
                        stats_json = :stats,
                        completed_at = :now
                    WHERE id = :aid
                    """
                ),
                {
                    "aid": analysis_id,
                    "graph": json.dumps(graph_payload),
                    "risk": json.dumps(risk_records),
                    "flags": flags_to_json(flags),
                    "stats": json.dumps(stats),
                    "now": datetime.now(timezone.utc),
                },
            )

            # Update dataset status
            session.execute(
                text("UPDATE datasets SET status = 'completed' WHERE id = :did"),
                {"did": dataset_id},
            )
            session.commit()

            logger.info("analysis_pipeline_complete", analysis_id=analysis_id, stats=stats)
            return {"status": "completed", "analysis_id": analysis_id, "stats": stats}

        except Exception as exc:
            session.rollback()
            session.execute(
                text(
                    """
                    UPDATE analysis_results
                    SET status = 'failed', error_message = :err
                    WHERE id = :aid
                    """
                ),
                {"aid": analysis_id, "err": str(exc)},
            )
            session.commit()
            logger.error("analysis_pipeline_failed", analysis_id=analysis_id, error=str(exc))
            raise


# ═══════════════════════════════════════════════════════════════
#  Task: Isomorphism Search
# ═══════════════════════════════════════════════════════════════

@celery_app.task(bind=True, name="analysis.run_isomorphism")
def run_isomorphism_search(
    self, analysis_id: str, dataset_id: str, target_node: str, hops: int
):
    """
    Run VF2 isomorphism search against the graph from a completed analysis.
    Updates the graph_json with match highlighting.
    """
    logger.info(
        "isomorphism_search_start",
        analysis_id=analysis_id,
        target_node=target_node,
        hops=hops,
    )

    with SyncSession() as session:
        try:
            # Load transactions and rebuild graph
            df = _load_transactions_df(session, dataset_id)
            G = nx.from_pandas_edgelist(
                df, "sender_id", "receiver_id", create_using=nx.DiGraph()
            )

            # Run isomorphism
            match_nodes, match_edges = find_structural_clones(G, target_node, hops)

            # Load existing analysis to get risk_df
            row = session.execute(
                text("SELECT risk_json FROM analysis_results WHERE id = :aid"),
                {"aid": analysis_id},
            ).fetchone()

            risk_records = json.loads(row[0]) if row and row[0] else []
            risk_df = pd.DataFrame(risk_records)

            # Rebuild graph payload with matches highlighted
            graph_payload = build_graph_payload(df, risk_df, match_nodes, match_edges)

            # Update analysis
            session.execute(
                text("UPDATE analysis_results SET graph_json = :graph WHERE id = :aid"),
                {"aid": analysis_id, "graph": json.dumps(graph_payload)},
            )
            session.commit()

            result = {
                "match_nodes": match_nodes,
                "match_edges": [[e[0], e[1]] for e in match_edges],
                "match_count": len(match_nodes),
            }

            logger.info("isomorphism_search_complete", **result)
            return result

        except Exception as exc:
            session.rollback()
            logger.error("isomorphism_search_failed", error=str(exc))
            raise
