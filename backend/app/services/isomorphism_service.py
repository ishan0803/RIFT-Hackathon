"""
VF2 Subgraph Isomorphism service — ported from the Streamlit prototype.

Finds all subgraphs in the network matching the exact topological shape
of a target node's neighbourhood. No size restrictions.
"""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import structlog

logger = structlog.get_logger(__name__)


def find_structural_clones(
    G: nx.DiGraph,
    target_node: str,
    hops: int,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Find all subgraphs in *G* that are isomorphic to the ego-graph
    of *target_node* within *hops* radius.

    Returns:
        match_nodes: list of node IDs that belong to matching subgraphs.
        match_edges: list of (source, target) edge tuples in matching subgraphs.
    """
    if target_node not in G:
        logger.warning("isomorphism_target_not_found", node=target_node)
        return [], []

    # 1. Extract reference shape
    ref_subgraph = nx.ego_graph(G, target_node, radius=hops, undirected=True)
    num_nodes = len(ref_subgraph.nodes())

    logger.info(
        "isomorphism_search_start",
        target_node=target_node,
        hops=hops,
        ref_subgraph_size=num_nodes,
        total_graph_nodes=G.number_of_nodes(),
    )

    target_in_deg = G.in_degree(target_node)
    target_out_deg = G.out_degree(target_node)

    match_nodes: set = set()
    match_edges: set = set()

    # 2. Hunt for matches (VF2 via is_isomorphic)
    for n in G.nodes():
        # Degree pre-filter: only check nodes with identical in/out degree
        if G.in_degree(n) == target_in_deg and G.out_degree(n) == target_out_deg:
            cand_subgraph = nx.ego_graph(G, n, radius=hops, undirected=True)

            if len(cand_subgraph.nodes()) == num_nodes:
                if nx.is_isomorphic(ref_subgraph, cand_subgraph):
                    match_nodes.update(cand_subgraph.nodes())
                    match_edges.update(cand_subgraph.edges())

    logger.info(
        "isomorphism_search_complete",
        match_node_count=len(match_nodes),
        match_edge_count=len(match_edges),
    )

    return list(match_nodes), [tuple(e) for e in match_edges]
