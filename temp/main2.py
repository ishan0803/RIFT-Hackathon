import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import streamlit.components.v1 as components
import uuid
import json
from datetime import datetime, timedelta
import random

st.set_page_config(layout="wide", page_title="AML Network Analyzer")

# --- 1. DATA GENERATION ---
@st.cache_data
def generate_synthetic_data(num_tx=10000):
    transactions = []
    base_time = datetime(2025, 1, 1)
    
    num_normal = int(num_tx * 0.95)
    normal_nodes = [f"ACC_{i:05d}" for i in range(1, 201)]
    
    for _ in range(num_normal):
        s, r = random.sample(normal_nodes, 2)
        t = base_time + timedelta(minutes=random.randint(0, 525600)) 
        transactions.append([str(uuid.uuid4()), s, r, round(random.uniform(10.0, 1000.0), 2), t])

    # Inject Typologies
    for i in range(20): 
        ring_size = random.randint(3, 5)
        ring_nodes = [f"RING_{i}_{j}" for j in range(ring_size)]
        for j in range(ring_size):
            s = ring_nodes[j]
            r = ring_nodes[(j + 1) % ring_size]
            t = base_time + timedelta(days=i, hours=j)
            transactions.append([str(uuid.uuid4()), s, r, round(random.uniform(5000, 10000), 2), t])

    for i in range(10): 
        aggregator = f"AGG_{i}"
        for j in range(15): 
            s = f"SMURF_S_{i}_{j}"
            t = base_time + timedelta(days=30+i, hours=random.randint(1, 70))
            transactions.append([str(uuid.uuid4()), s, aggregator, round(random.uniform(1000, 3000), 2), t])
            
    for i in range(10): 
        disperser = f"DISP_{i}"
        for j in range(15): 
            r = f"SMURF_R_{i}_{j}"
            t = base_time + timedelta(days=60+i, hours=random.randint(1, 70))
            transactions.append([str(uuid.uuid4()), disperser, r, round(random.uniform(1000, 3000), 2), t])

    for i in range(20): 
        start_node = f"SHELL_START_{i}"
        end_node = f"SHELL_END_{i}"
        shell_1 = f"SHELL_MID1_{i}"
        shell_2 = f"SHELL_MID2_{i}"
        
        t1 = base_time + timedelta(days=90+i, hours=1)
        t2 = t1 + timedelta(hours=2)
        t3 = t2 + timedelta(hours=2)
        
        transactions.append([str(uuid.uuid4()), start_node, shell_1, 50000.0, t1])
        transactions.append([str(uuid.uuid4()), shell_1, shell_2, 49950.0, t2])
        transactions.append([str(uuid.uuid4()), shell_2, end_node, 49900.0, t3])

    df = pd.DataFrame(transactions, columns=['transaction_id', 'sender_id', 'receiver_id', 'amount', 'timestamp'])
    return df.sample(frac=1).reset_index(drop=True)

# --- 2. FAST PATTERN DETECTION ---
def analyze_networks(df):
    flags = {'cycles': set(), 'fan_in': set(), 'fan_out': set(), 'shells': set()}
    
    df_sorted = df.sort_values('timestamp')
    td_72 = pd.Timedelta(hours=72)
    
    for receiver, group in df_sorted.groupby('receiver_id'):
        if len(group) >= 10:
            diffs = group['timestamp'].diff(periods=9) 
            if (diffs <= td_72).any():
                flags['fan_in'].add(receiver)
                
    for sender, group in df_sorted.groupby('sender_id'):
        if len(group) >= 10:
            diffs = group['timestamp'].diff(periods=9)
            if (diffs <= td_72).any():
                flags['fan_out'].add(sender)

    G = nx.from_pandas_edgelist(df, 'sender_id', 'receiver_id', create_using=nx.DiGraph())
    
    suspect_cycle_nodes = set()
    max_depth = 5
    for start_node in G.nodes():
        if start_node in suspect_cycle_nodes: continue 
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
    flags['cycles'].update(suspect_cycle_nodes)

    all_nodes = pd.concat([df['sender_id'], df['receiver_id']])
    node_counts = all_nodes.value_counts()
    shell_candidates = set(node_counts[(node_counts >= 2) & (node_counts <= 3)].index)
    
    for node in shell_candidates:
        if node not in G: continue
        successors = set(G.successors(node))
        if successors.intersection(shell_candidates):
            flags['shells'].add(node)
            flags['shells'].update(successors.intersection(shell_candidates))

    return G, flags

# --- 3. RISK SCORING ---
def assign_risk_scores(nodes, flags):
    risk_scores = {}
    for node in nodes:
        score = 0
        reasons = []
        if node in flags['cycles']:
            score += 40; reasons.append("Cycle (Ring)")
        if node in flags['fan_in']:
            score += 35; reasons.append("Fan-in")
        if node in flags['fan_out']:
            score += 35; reasons.append("Fan-out")
        if node in flags['shells']:
            score += 25; reasons.append("Shell Layer")
            
        risk_scores[node] = {
            'score': min(score, 100), 
            'risk_level': 'High' if score >= 40 else 'Medium' if score > 0 else 'Low',
            'reasons': ", ".join(reasons) if reasons else "Normal"
        }
    return pd.DataFrame.from_dict(risk_scores, orient='index').reset_index().rename(columns={'index': 'account_id'})

# --- 4. SUBGRAPH ISOMORPHISM (PATTERN MATCHING) ---
def find_structural_clones(G, target_node, hops):
    """Finds all subgraphs in the network that match the exact shape of the target's subgraph."""
    if target_node not in G:
        return [], []
        
    # 1. Extract the reference shape 
    ref_subgraph = nx.ego_graph(G, target_node, radius=hops, undirected=True)
    num_nodes = len(ref_subgraph.nodes())
    
    # Safety rails removed! We will log the size so you know how hard the CPU is working.
    st.sidebar.info(f"🧠 Target subgraph has {num_nodes} nodes. Searching entire network for exact matches...")
        
    match_nodes = set()
    match_edges = set()
    
    target_in_deg = G.in_degree(target_node)
    target_out_deg = G.out_degree(target_node)
    
    # 2. Hunt for matches (VF2 Algorithm via is_isomorphic)
    for n in G.nodes():
        # Optimization: Only check nodes with the exact same incoming/outgoing transaction counts
        if G.in_degree(n) == target_in_deg and G.out_degree(n) == target_out_deg:
            
            cand_subgraph = nx.ego_graph(G, n, radius=hops, undirected=True)
            
            # Subgraphs must have the exact same number of nodes to be identical
            if len(cand_subgraph.nodes()) == num_nodes:
                if nx.is_isomorphic(ref_subgraph, cand_subgraph):
                    match_nodes.update(cand_subgraph.nodes())
                    match_edges.update(cand_subgraph.edges())
                    
    return list(match_nodes), list(match_edges)

# --- 5. INTERACTIVE D3 CANVAS VISUALIZATION ---
def create_interactive_d3_canvas(df, risk_df, match_nodes=[], match_edges=[]):
    risk_lookup = risk_df.set_index('account_id').to_dict('index')
    all_nodes = set(df['sender_id']).union(set(df['receiver_id']))
    
    # Convert match_edges tuples to a set of strings for fast JS lookup
    match_edges_set = {f"{u}->{v}" for u, v in match_edges}
    
    nodes_data = []
    for n in all_nodes:
        info = risk_lookup.get(n, {'score': 0, 'risk_level': 'Low', 'reasons': 'Normal'})
        color = "#ff4b4b" if info['risk_level'] == 'High' else "#ffa500" if info['risk_level'] == 'Medium' else "#1f77b4"
        is_match = 1 if n in match_nodes else 0
        radius = 8 if (info['score'] > 0 or is_match) else 3.5
        
        nodes_data.append({
            "id": n, "color": color, "radius": radius, "is_match": is_match,
            "title": f"<b>{n}</b><br/>Risk Score: {info['score']}<br/>Flags: {info['reasons']}"
        })

    links_data = []
    for _, row in df.iterrows():
        is_match = 1 if f"{row['sender_id']}->{row['receiver_id']}" in match_edges_set else 0
        links_data.append({"source": row['sender_id'], "target": row['receiver_id'], "is_match": is_match})

    graph_data = json.dumps({"nodes": nodes_data, "links": links_data})

    html_content = f"""
    <!DOCTYPE html>
    <meta charset="utf-8">
    <title>Interactive D3 Canvas</title>
    <style>
        body {{ margin: 0; background-color: #1e1e1e; overflow: hidden; font-family: sans-serif; }}
        canvas {{ display: block; cursor: grab; }}
        canvas:active {{ cursor: grabbing; }}
        #tooltip {{ position: absolute; opacity: 0; background: rgba(20, 20, 20, 0.9); padding: 10px; border-radius: 6px; pointer-events: none; font-size: 13px; color: white; border: 1px solid #444; z-index: 10; }}
    </style>
    <body>
        <div id="tooltip"></div>
        <canvas id="network"></canvas>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <script>
            const data = {graph_data};
            const canvas = document.getElementById("network");
            const context = canvas.getContext("2d");
            const tooltip = document.getElementById("tooltip");

            let width = window.innerWidth;
            let height = 650;
            const dpr = window.devicePixelRatio || 1;
            canvas.width = width * dpr; canvas.height = height * dpr;
            context.scale(dpr, dpr); canvas.style.width = width + "px"; canvas.style.height = height + "px";

            let transform = d3.zoomIdentity;
            let hoveredNode = null;

            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links).id(d => d.id).distance(25))
                .force("charge", d3.forceManyBody().strength(-20))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collide", d3.forceCollide().radius(d => d.radius + 2));

            d3.select(canvas).call(d3.zoom().scaleExtent([0.05, 10]).on("zoom", (e) => {{ transform = e.transform; draw(); }}));

            d3.select(canvas).call(d3.drag()
                .subject((e) => simulation.find(transform.invertX(e.x), transform.invertY(e.y), 20))
                .on("start", (e) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); e.subject.fx = e.subject.x; e.subject.fy = e.subject.y; }})
                .on("drag", (e) => {{ e.subject.fx = transform.invertX(e.x); e.subject.fy = transform.invertY(e.y); }})
                .on("end", (e) => {{ if (!e.active) simulation.alphaTarget(0); e.subject.fx = null; e.subject.fy = null; }})
            );

            d3.select(canvas).on("mousemove", (e) => {{
                const node = simulation.find(transform.invertX(e.offsetX), transform.invertY(e.offsetY), 10); 
                if (node !== hoveredNode) {{
                    hoveredNode = node;
                    if (node) {{ tooltip.style.opacity = 1; tooltip.innerHTML = node.title; }} 
                    else {{ tooltip.style.opacity = 0; }}
                    draw(); 
                }}
                if (node) {{ tooltip.style.left = (e.pageX + 15) + "px"; tooltip.style.top = (e.pageY + 15) + "px"; }}
            }});

            simulation.on("tick", draw);

            function draw() {{
                context.save();
                context.clearRect(0, 0, width, height);
                context.translate(transform.x, transform.y);
                context.scale(transform.k, transform.k);

                // Draw Normal Links
                context.beginPath();
                data.links.forEach(d => {{ if(d.is_match === 0) {{ context.moveTo(d.source.x, d.source.y); context.lineTo(d.target.x, d.target.y); }} }});
                context.strokeStyle = "rgba(100, 100, 100, 0.15)";
                context.lineWidth = 0.5 / transform.k; 
                context.stroke();

                // Draw Hover Links (White)
                if (hoveredNode) {{
                    context.beginPath();
                    data.links.forEach(d => {{ if (d.source === hoveredNode || d.target === hoveredNode) {{ context.moveTo(d.source.x, d.source.y); context.lineTo(d.target.x, d.target.y); }} }});
                    context.strokeStyle = "rgba(255, 255, 255, 0.9)";
                    context.lineWidth = 2.0 / transform.k;
                    context.stroke();
                }}

                // Draw Pattern Matched Links (Neon Green)
                context.beginPath();
                data.links.forEach(d => {{ if(d.is_match === 1) {{ context.moveTo(d.source.x, d.source.y); context.lineTo(d.target.x, d.target.y); }} }});
                context.strokeStyle = "#00ff00"; // Neon Green
                context.lineWidth = 2.5 / transform.k; 
                context.stroke();

                // Draw Nodes
                data.nodes.forEach(d => {{
                    context.beginPath();
                    context.moveTo(d.x + d.radius, d.y);
                    context.arc(d.x, d.y, d.radius, 0, 2 * Math.PI);
                    context.fillStyle = d.color;
                    context.fill();

                    // Stroke for match or hover
                    if (d.is_match === 1 || d === hoveredNode) {{
                        context.strokeStyle = d === hoveredNode ? "white" : "#00ff00";
                        context.lineWidth = 3 / transform.k;
                        context.stroke();
                    }}
                }});
                context.restore();
            }}
        </script>
    </body>
    </html>
    """
    return html_content

# --- UI LAYOUT ---
st.title("🛡️ Anti-Money Laundering (AML) Network Analyzer")

col1, col2 = st.sidebar.columns(2)
num_transactions = st.sidebar.slider("Number of Transactions", 1000, 20000, 10000, 1000)

if 'df' not in st.session_state:
    st.session_state.df = generate_synthetic_data(10000)
    st.session_state.G, st.session_state.flags = analyze_networks(st.session_state.df)
    st.session_state.risk_df = assign_risk_scores(list(st.session_state.G.nodes()), st.session_state.flags)

if st.sidebar.button("Generate & Analyze Data", type="primary"):
    with st.spinner("Processing..."):
        st.session_state.df = generate_synthetic_data(num_transactions)
        st.session_state.G, st.session_state.flags = analyze_networks(st.session_state.df)
        st.session_state.risk_df = assign_risk_scores(list(st.session_state.G.nodes()), st.session_state.flags)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Subgraph Pattern Hunter")
st.sidebar.markdown("Find networks with identical topological structures.")
target_node_input = st.sidebar.text_input("Target Node ID (e.g., AGG_0)")
hop_count = st.sidebar.number_input("Depth (Hops)", min_value=1, max_value=10, value=1, help="Restricted to 2 to prevent memory crash.")

match_nodes, match_edges = [], []
if st.sidebar.button("Hunt for Clones"):
    if target_node_input in st.session_state.G:
        with st.spinner("Running VF2 Isomorphism algorithm..."):
            match_nodes, match_edges = find_structural_clones(st.session_state.G, target_node_input, hop_count)
        st.sidebar.success(f"Found {len(match_nodes)} nodes matching this structure!")
    else:
        st.sidebar.error("Node not found in graph.")

tab1, tab2, tab3 = st.tabs(["📊 Network Visualization", "🚩 Risk Intelligence", "🗄️ Raw Data"])

with tab1:
    st.subheader("Interactive Suspicious Actor Network")
    st.markdown("*Use the sidebar to hunt for structural clones. Matches will illuminate in **Neon Green**.*")
    d3_html = create_interactive_d3_canvas(st.session_state.df, st.session_state.risk_df, match_nodes, match_edges)
    components.html(d3_html, height=660)

with tab2:
    st.dataframe(st.session_state.risk_df[st.session_state.risk_df['score'] > 0].sort_values('score', ascending=False), use_container_width=True)

with tab3:
    st.dataframe(st.session_state.df, use_container_width=True)