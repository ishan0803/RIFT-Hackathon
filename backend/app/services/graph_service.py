"""
Graph analysis service — ported from the Streamlit prototype.

Contains cycle detection (DFS), smurfing (fan-in / fan-out 72h window),
layered shell detection, risk scoring, and D3 graph payload construction.

IMPORTANT: The core detection logic (analyze_networks, assign_risk_scores)
is an exact 1:1 port of the Streamlit app — do NOT modify the algorithms.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Pattern Detection  (EXACT copy of Streamlit logic)
# ═══════════════════════════════════════════════════════════════

def analyze_networks(df: pd.DataFrame) -> Tuple[nx.DiGraph, Dict[str, Set[str]]]:
    """
    Run all AML pattern detectors on a transaction DataFrame.

    Returns:
        G:     Directed graph built from the transaction data.
        flags: Dict of sets — keys are ``cycles``, ``fan_in``, ``fan_out``, ``shells``.
    """
    flags: Dict[str, Set[str]] = {
        "cycles": set(),
        "fan_in": set(),
        "fan_out": set(),
        "shells": set(),
    }

    # ── 1. Smurfing Detection (fan-in / fan-out within 72 hours) ──
    df_sorted = df.sort_values("timestamp")
    td_72 = pd.Timedelta(hours=72)

    for receiver, group in df_sorted.groupby("receiver_id"):
        if len(group) >= 10:
            diffs = group["timestamp"].diff(periods=9)
            if (diffs <= td_72).any():
                flags["fan_in"].add(receiver)

    for sender, group in df_sorted.groupby("sender_id"):
        if len(group) >= 10:
            diffs = group["timestamp"].diff(periods=9)
            if (diffs <= td_72).any():
                flags["fan_out"].add(sender)

    # ── 2. Build Graph ────────────────────────────────────────────
    G = nx.from_pandas_edgelist(
        df, "sender_id", "receiver_id", create_using=nx.DiGraph()
    )

    # ── 3. Cycle Detection (Custom DFS, depth 3-5) ───────────────
    suspect_cycle_nodes: Set[str] = set()
    max_depth = 5

    for start_node in G.nodes():
        if start_node in suspect_cycle_nodes:
            continue

        stack = [(start_node, {start_node}, 1)]
        found_cycle = False

        while stack and not found_cycle:
            curr, path_nodes, depth = stack.pop()

            for neighbor in G.successors(curr):
                if neighbor == start_node and depth >= 3:
                    suspect_cycle_nodes.update(path_nodes)
                    found_cycle = True
                    break

                if neighbor not in path_nodes and depth < max_depth:
                    new_path = path_nodes.copy()
                    new_path.add(neighbor)
                    stack.append((neighbor, new_path, depth + 1))

    flags["cycles"].update(suspect_cycle_nodes)

    # ── 4. Layered Shell Detection ────────────────────────────────
    all_nodes = pd.concat([df["sender_id"], df["receiver_id"]])
    node_counts = all_nodes.value_counts()
    shell_candidates = set(
        node_counts[(node_counts >= 2) & (node_counts <= 3)].index
    )

    for node in shell_candidates:
        if node not in G:
            continue
        successors = set(G.successors(node))
        if successors.intersection(shell_candidates):
            flags["shells"].add(node)
            flags["shells"].update(successors.intersection(shell_candidates))

    logger.info(
        "analysis_complete",
        nodes=G.number_of_nodes(),
        edges=G.number_of_edges(),
        cycles=len(flags["cycles"]),
        fan_in=len(flags["fan_in"]),
        fan_out=len(flags["fan_out"]),
        shells=len(flags["shells"]),
    )

    return G, flags


# ═══════════════════════════════════════════════════════════════
#  Risk Scoring  (EXACT copy of Streamlit logic)
# ═══════════════════════════════════════════════════════════════

def assign_risk_scores(
    nodes: List[str], flags: Dict[str, Set[str]]
) -> pd.DataFrame:
    """Assign 0-100 risk scores to each node based on detected patterns."""
    # Handle empty graph gracefully
    if not nodes:
        return pd.DataFrame(
            columns=["account_id", "score", "risk_level", "reasons"]
        )

    risk_scores = {}

    for node in nodes:
        score = 0
        reasons = []

        if node in flags["cycles"]:
            score += 40
            reasons.append("Cycle (Ring)")
        if node in flags["fan_in"]:
            score += 35
            reasons.append("Fan-in (Aggregator)")
        if node in flags["fan_out"]:
            score += 35
            reasons.append("Fan-out (Disperser)")
        if node in flags["shells"]:
            score += 25
            reasons.append("Shell Layer")

        risk_scores[node] = {
            "score": min(score, 100),
            "risk_level": (
                "High" if score >= 40 else "Medium" if score > 0 else "Low"
            ),
            "reasons": ", ".join(reasons) if reasons else "Normal",
        }

    return (
        pd.DataFrame.from_dict(risk_scores, orient="index")
        .reset_index()
        .rename(columns={"index": "account_id"})
    )


# ═══════════════════════════════════════════════════════════════
#  D3 Graph Payload Builder
# ═══════════════════════════════════════════════════════════════

def build_graph_payload(
    df: pd.DataFrame,
    risk_df: pd.DataFrame,
    match_nodes: Optional[List[str]] = None,
    match_edges: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable graph payload for D3 visualization.

    Returns dict with ``nodes`` and ``links`` lists.
    """
    match_nodes = set(match_nodes or [])
    # Normalize match_edges: accept list of tuples, lists, or any iterable pairs
    _raw_edges = match_edges or []
    match_edges_set = set()
    for edge in _raw_edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            match_edges_set.add(f"{edge[0]}->{edge[1]}")
        elif isinstance(edge, str):
            match_edges_set.add(edge)

    risk_lookup = risk_df.set_index("account_id").to_dict("index")
    all_nodes = set(df["sender_id"]).union(set(df["receiver_id"]))

    nodes_data = []
    for n in all_nodes:
        info = risk_lookup.get(
            n, {"score": 0, "risk_level": "Low", "reasons": "Normal"}
        )
        color = (
            "#ff4b4b"
            if info["risk_level"] == "High"
            else "#ffa500"
            if info["risk_level"] == "Medium"
            else "#1f77b4"
        )
        is_match = 1 if n in match_nodes else 0
        radius = 8 if (info["score"] > 0 or is_match) else 3.5

        nodes_data.append(
            {
                "id": n,
                "color": color,
                "radius": radius,
                "is_match": is_match,
                "title": (
                    f"<b>{n}</b><br/>Risk Score: {info['score']}<br/>"
                    f"Flags: {info['reasons']}"
                ),
            }
        )

    links_data = []
    seen_links = set()
    for _, row in df.iterrows():
        link_key = f"{row['sender_id']}->{row['receiver_id']}"
        if link_key not in seen_links:
            seen_links.add(link_key)
            is_match = 1 if link_key in match_edges_set else 0
            links_data.append(
                {
                    "source": row["sender_id"],
                    "target": row["receiver_id"],
                    "is_match": is_match,
                }
            )

    return {"nodes": nodes_data, "links": links_data}


# ═══════════════════════════════════════════════════════════════
#  Structured Output Builder
# ═══════════════════════════════════════════════════════════════

def build_structured_output(
    G: nx.DiGraph,
    flags: Dict[str, Set[str]],
    risk_df: pd.DataFrame,
    processing_time: float,
) -> Dict[str, Any]:
    """
    Build the structured output JSON with:
      - suspicious_accounts
      - fraud_rings
      - summary
    """
    risk_lookup = risk_df.set_index("account_id").to_dict("index")

    # ── Build Fraud Rings ─────────────────────────────────────
    fraud_rings: List[Dict[str, Any]] = []
    ring_counter = 0
    account_ring_map: Dict[str, str] = {}  # account_id → ring_id

    # 1. Cycle-based rings: find connected components among cycle nodes
    if flags["cycles"]:
        cycle_subgraph = G.subgraph(flags["cycles"]).copy()
        for component in nx.weakly_connected_components(cycle_subgraph):
            if len(component) >= 2:
                ring_counter += 1
                ring_id = f"RING_{ring_counter:03d}"
                members = sorted(component)

                # Compute average risk score for ring
                ring_scores = [
                    risk_lookup.get(m, {}).get("score", 0) for m in members
                ]
                avg_score = round(
                    sum(ring_scores) / len(ring_scores), 1
                ) if ring_scores else 0.0

                fraud_rings.append({
                    "ring_id": ring_id,
                    "member_accounts": members,
                    "pattern_type": "cycle",
                    "risk_score": avg_score,
                })
                for m in members:
                    account_ring_map[m] = ring_id

    # 2. Fan-in clusters (aggregator rings)
    for agg_node in sorted(flags["fan_in"]):
        predecessors = set(G.predecessors(agg_node))
        cluster = {agg_node} | predecessors
        if len(cluster) >= 3 and agg_node not in account_ring_map:
            ring_counter += 1
            ring_id = f"RING_{ring_counter:03d}"
            members = sorted(cluster)
            ring_scores = [
                risk_lookup.get(m, {}).get("score", 0) for m in members
            ]
            avg_score = round(
                sum(ring_scores) / len(ring_scores), 1
            ) if ring_scores else 0.0

            fraud_rings.append({
                "ring_id": ring_id,
                "member_accounts": members,
                "pattern_type": "fan_in",
                "risk_score": avg_score,
            })
            for m in members:
                if m not in account_ring_map:
                    account_ring_map[m] = ring_id

    # 3. Fan-out clusters (disperser rings)
    for disp_node in sorted(flags["fan_out"]):
        successors = set(G.successors(disp_node))
        cluster = {disp_node} | successors
        if len(cluster) >= 3 and disp_node not in account_ring_map:
            ring_counter += 1
            ring_id = f"RING_{ring_counter:03d}"
            members = sorted(cluster)
            ring_scores = [
                risk_lookup.get(m, {}).get("score", 0) for m in members
            ]
            avg_score = round(
                sum(ring_scores) / len(ring_scores), 1
            ) if ring_scores else 0.0

            fraud_rings.append({
                "ring_id": ring_id,
                "member_accounts": members,
                "pattern_type": "fan_out",
                "risk_score": avg_score,
            })
            for m in members:
                if m not in account_ring_map:
                    account_ring_map[m] = ring_id

    # 4. Shell layer chains
    if flags["shells"]:
        shell_subgraph = G.subgraph(flags["shells"]).copy()
        for component in nx.weakly_connected_components(shell_subgraph):
            if len(component) >= 2:
                ring_counter += 1
                ring_id = f"RING_{ring_counter:03d}"
                members = sorted(component)
                ring_scores = [
                    risk_lookup.get(m, {}).get("score", 0) for m in members
                ]
                avg_score = round(
                    sum(ring_scores) / len(ring_scores), 1
                ) if ring_scores else 0.0

                fraud_rings.append({
                    "ring_id": ring_id,
                    "member_accounts": members,
                    "pattern_type": "shell_layering",
                    "risk_score": avg_score,
                })
                for m in members:
                    if m not in account_ring_map:
                        account_ring_map[m] = ring_id

    # ── Build Suspicious Accounts ─────────────────────────────
    suspicious_accounts: List[Dict[str, Any]] = []

    for _, row in risk_df.iterrows():
        if row["score"] <= 0:
            continue

        account_id = row["account_id"]
        patterns = []
        if account_id in flags["cycles"]:
            # Determine cycle length
            if account_id in G:
                try:
                    for cycle in nx.simple_cycles(
                        G.subgraph(flags["cycles"]), length_bound=6
                    ):
                        if account_id in cycle:
                            patterns.append(f"cycle_length_{len(cycle)}")
                            break
                except Exception:
                    patterns.append("cycle")
            if not any(p.startswith("cycle") for p in patterns):
                patterns.append("cycle")
        if account_id in flags["fan_in"]:
            patterns.append("high_velocity")
            patterns.append("fan_in_aggregator")
        if account_id in flags["fan_out"]:
            patterns.append("high_velocity")
            patterns.append("fan_out_disperser")
        if account_id in flags["shells"]:
            patterns.append("shell_layer")

        # Deduplicate
        patterns = list(dict.fromkeys(patterns))

        suspicious_accounts.append({
            "account_id": account_id,
            "suspicion_score": round(float(row["score"]), 1),
            "detected_patterns": patterns,
            "ring_id": account_ring_map.get(account_id),
        })

    # Sort by score descending
    suspicious_accounts.sort(key=lambda x: x["suspicion_score"], reverse=True)

    # ── Build Summary ─────────────────────────────────────────
    summary = {
        "total_accounts_analyzed": G.number_of_nodes(),
        "suspicious_accounts_flagged": len(suspicious_accounts),
        "fraud_rings_detected": len(fraud_rings),
        "processing_time_seconds": round(processing_time, 2),
    }

    return {
        "suspicious_accounts": suspicious_accounts,
        "fraud_rings": fraud_rings,
        "summary": summary,
    }


# ═══════════════════════════════════════════════════════════════
#  Serialized Flags (for DB storage)
# ═══════════════════════════════════════════════════════════════

def flags_to_json(flags: Dict[str, Set[str]]) -> str:
    """Convert flags dict (with sets) to JSON string."""
    return json.dumps({k: list(v) for k, v in flags.items()})


def flags_from_json(raw: str) -> Dict[str, Set[str]]:
    """Restore flags dict from JSON string."""
    data = json.loads(raw)
    return {k: set(v) for k, v in data.items()}
