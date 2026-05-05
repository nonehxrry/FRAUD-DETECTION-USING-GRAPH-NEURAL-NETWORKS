"""
utils/visualization.py
-----------------------
Graph visualization and fraud cluster analysis using NetworkX & PyVis.
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
import argparse


def build_nx_subgraph(df: pd.DataFrame,
                       fraud_probs: np.ndarray,
                       n_transactions: int = 200,
                       prob_threshold: float = 0.5) -> nx.Graph:
    """
    Build a small NetworkX graph for visualization.
    Only includes transactions above the probability threshold
    and their immediate card/device/email neighbours.
    """
    G = nx.Graph()

    # Sample high-risk transactions
    idx = np.where(fraud_probs >= prob_threshold)[0]
    idx = idx[:n_transactions]  # cap for performance

    for i in idx:
        row = df.iloc[i]
        tx_id = f"TX_{i}"
        prob  = float(fraud_probs[i])
        label = "FRAUD" if row.get('isFraud', 0) == 1 else "SUSPECTED"
        G.add_node(tx_id, node_type='transaction',
                   fraud_prob=prob, label=label, color='#FF4B4B' if prob > 0.7 else '#FFA500')

        # Card node
        if 'card1' in row:
            card_id = f"CARD_{int(row['card1'])}"
            G.add_node(card_id, node_type='card', color='#4B9EFF', label='card')
            G.add_edge(tx_id, card_id, weight=1.0)

        # Email node
        if 'P_emaildomain' in row:
            email_val = row['P_emaildomain']
            if pd.notna(email_val):
                email_id = f"EMAIL_{str(email_val)}"
                G.add_node(email_id, node_type='email', color='#4BFF9E', label='email')
                G.add_edge(tx_id, email_id, weight=0.8)

        # Device node
        if 'DeviceType' in row:
            dev_val = row['DeviceType']
            if pd.notna(dev_val):
                dev_id = f"DEV_{str(dev_val)}"
                G.add_node(dev_id, node_type='device', color='#FF4BFF', label='device')
                G.add_edge(tx_id, dev_id, weight=0.9)

    return G


def plot_fraud_distribution(df: pd.DataFrame) -> go.Figure:
    """Plotly chart: fraud vs non-fraud counts + TransactionAmt distribution."""
    fig = go.Figure()
    counts = df['isFraud'].value_counts().reset_index()
    counts.columns = ['isFraud', 'count']
    counts['label'] = counts['isFraud'].map({0: 'Legitimate', 1: 'Fraudulent'})

    fig.add_trace(go.Bar(
        x=counts['label'],
        y=counts['count'],
        marker_color=['#00CC96', '#FF4B4B'],
        text=counts['count'],
        textposition='outside'
    ))
    fig.update_layout(
        title='Transaction Class Distribution',
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font_color='white',
        showlegend=False
    )
    return fig


def plot_fraud_amount_dist(df: pd.DataFrame) -> go.Figure:
    """Transaction amount distribution by fraud label."""
    legit = df[df['isFraud'] == 0]['TransactionAmt']
    fraud = df[df['isFraud'] == 1]['TransactionAmt']

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=legit, name='Legitimate',
                               marker_color='#00CC96', opacity=0.7,
                               xbins=dict(size=0.1)))
    fig.add_trace(go.Histogram(x=fraud, name='Fraudulent',
                               marker_color='#FF4B4B', opacity=0.7,
                               xbins=dict(size=0.1)))
    fig.update_layout(
        title='Transaction Amount Distribution (log scale)',
        barmode='overlay',
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font_color='white',
        xaxis_title='log(1 + TransactionAmt)',
        yaxis_title='Count'
    )
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
    """Plotly ROC curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'GNN (AUC={auc:.4f})',
                             line=dict(color='#4B9EFF', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             name='Random', line=dict(color='gray', dash='dash')))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font_color='white'
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Plotly confusion matrix heatmap."""
    labels = ['Legitimate', 'Fraudulent']
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='RdBu',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 18}
    ))
    fig.update_layout(
        title='Confusion Matrix',
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font_color='white',
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    return fig


def generate_pyvis_html(G: nx.Graph, output_path: str = "fraud_graph.html"):
    """
    Generate an interactive PyVis graph HTML file.
    Color-coded: red=fraud, orange=suspected, blue=card, green=email, purple=device
    """
    net = Network(height="600px", width="100%",
                  bgcolor="#0E1117", font_color="white",
                  notebook=False)

    for node, attrs in G.nodes(data=True):
        color = attrs.get('color', '#888888')
        title = f"{node} | Type: {attrs.get('node_type','?')}"
        if 'fraud_prob' in attrs:
            title += f" | Fraud Prob: {attrs['fraud_prob']:.3f}"
        net.add_node(node, label=str(node)[:12], color=color,
                     title=title, size=15)

    for src, dst, attrs in G.edges(data=True):
        net.add_edge(src, dst, color='#555555')

    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "springLength": 100
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    net.save_graph(output_path)
    return output_path


def plot_feature_importance(importance_dict: dict, top_n: int = 20) -> go.Figure:
    """Bar chart of feature importance scores."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, scores = zip(*sorted_items)

    fig = go.Figure(go.Bar(
        x=list(scores)[::-1],
        y=list(features)[::-1],
        orientation='h',
        marker_color='#4B9EFF'
    ))
    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance Score',
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font_color='white',
        height=600
    )
    return fig


def _resolve_default_transaction_path() -> Path:
    """Prefer demo CSV for quick local rendering, else fall back to full train CSV."""
    candidates = [
        Path("data/demo_transaction.csv"),
        Path("data/train_transaction.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No transaction CSV found in data/.")


def _resolve_default_probs_path() -> Path:
    probs_path = Path("models/saved/test_probs.npy")
    if probs_path.exists():
        return probs_path
    raise FileNotFoundError("models/saved/test_probs.npy not found. Train a model first.")


def main():
    parser = argparse.ArgumentParser(description="Generate interactive PyVis fraud graph HTML.")
    parser.add_argument("--transaction", type=str, default=None,
                        help="Path to transaction CSV (default: data/demo_transaction.csv or data/train_transaction.csv)")
    parser.add_argument("--probs", type=str, default=None,
                        help="Path to fraud probability .npy file (default: models/saved/test_probs.npy)")
    parser.add_argument("--output", type=str, default="fraud_graph.html",
                        help="Output HTML path")
    parser.add_argument("--n_transactions", type=int, default=200,
                        help="Maximum number of high-risk transactions to include")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fraud probability threshold")
    args = parser.parse_args()

    tx_path = Path(args.transaction) if args.transaction else _resolve_default_transaction_path()
    probs_path = Path(args.probs) if args.probs else _resolve_default_probs_path()

    df = pd.read_csv(tx_path)
    probs = np.load(probs_path)

    # Align lengths safely when CSV has more rows than saved test probabilities.
    n = min(len(df), len(probs))
    df = df.iloc[:n].reset_index(drop=True)
    probs = probs[:n]

    G = build_nx_subgraph(
        df=df,
        fraud_probs=probs,
        n_transactions=args.n_transactions,
        prob_threshold=args.threshold,
    )
    output_path = generate_pyvis_html(G, output_path=args.output)
    print(f"Generated PyVis graph: {output_path}")


if __name__ == "__main__":
    main()
