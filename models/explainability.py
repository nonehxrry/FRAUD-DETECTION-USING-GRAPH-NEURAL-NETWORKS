"""
models/explainability.py
------------------------
GNN Explainability using GNNExplainer + Feature Importance.

Two approaches:
1. GNNExplainer (local): explains individual predictions
   → "Why was THIS transaction flagged?"

2. Gradient-based global importance: which features matter most overall?
   → "What patterns does the model look for?"

3. Fraud cluster detection via community detection on the graph.
"""

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional


def compute_gradient_feature_importance(
        model,
        data: HeteroData,
        mask: torch.Tensor,
        feature_names: List[str],
        n_samples: int = 500,
        device: str = 'cpu') -> Dict[str, float]:
    """
    Gradient × Input feature importance.

    For each transaction node in the mask:
      importance_i = mean |grad(output_fraud_class) / grad(x_i)| × |x_i|

    This tells us which input features most influenced fraud predictions.

    Parameters
    ----------
    model         : trained FraudGNN
    data          : HeteroData graph
    mask          : boolean mask for transactions to explain
    feature_names : list of feature column names
    n_samples     : subsample for speed

    Returns
    -------
    dict mapping feature_name → importance_score (normalized 0-1)
    """
    model.eval()
    model.to(device)

    # Subsample for speed
    idx = torch.where(mask)[0][:n_samples]

    x_tx = data['transaction'].x[idx].to(device).requires_grad_(True)

    # Forward pass
    x_dict_copy = {nt: data[nt].x.to(device) for nt in data.node_types}
    x_dict_copy['transaction'] = x_tx

    out = model(x_dict_copy, {et: ei.to(device)
                              for et, ei in data.edge_index_dict.items()})

    # Backprop w.r.t. fraud class (index 1)
    fraud_logits = out[:len(idx), 1].sum()
    fraud_logits.backward()

    grads = x_tx.grad.detach().cpu().abs()         # [n_samples, n_feats]
    vals  = x_tx.detach().cpu().abs()

    importance = (grads * vals).mean(dim=0).numpy()  # [n_feats]

    # Normalize
    importance = importance / (importance.sum() + 1e-8)

    result = {name: float(score)
              for name, score in zip(feature_names[:len(importance)], importance)}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def explain_single_transaction(
        model,
        data: HeteroData,
        tx_idx: int,
        device: str = 'cpu',
        epochs: int = 200) -> dict:
    """
    Use GNNExplainer to explain a single transaction prediction.

    Returns a dict with:
        - node_mask: importance of neighboring nodes
        - edge_mask: importance of each edge
        - fraud_prob: model's fraud probability
        - prediction: 0 or 1
    """
    model.eval()
    model.to(device)

    try:
        # GNNExplainer is available in PyG ≥ 2.3
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(mode='multiclass_classification',
                              task_level='node',
                              return_type='raw')
        )
        explanation = explainer(
            x=data['transaction'].x.to(device),
            edge_index=data['transaction', 'same_card', 'transaction'
                           ].edge_index.to(device),
            index=tx_idx
        )
        node_mask = explanation.node_mask.cpu().numpy()
        edge_mask = explanation.edge_mask.cpu().numpy() if explanation.edge_mask is not None else None

    except Exception as e:
        # Fallback: gradient-only explanation
        node_mask = None
        edge_mask = None

    with torch.no_grad():
        out = model(
            {nt: data[nt].x.to(device) for nt in data.node_types},
            {et: ei.to(device) for et, ei in data.edge_index_dict.items()}
        )
        probs = F.softmax(out[tx_idx], dim=-1).cpu().numpy()

    return {
        'fraud_prob' : float(probs[1]),
        'prediction' : int(probs[1] > 0.5),
        'node_mask'  : node_mask,
        'edge_mask'  : edge_mask,
        'confidence' : float(max(probs))
    }


def detect_fraud_clusters(G: nx.Graph,
                           fraud_probs: np.ndarray,
                           threshold: float = 0.5) -> List[List[str]]:
    """
    Detect fraud clusters using Louvain community detection.
    Clusters with high average fraud probability are flagged.

    Parameters
    ----------
    G            : NetworkX graph (from build_nx_subgraph)
    fraud_probs  : array of fraud probabilities for transaction nodes
    threshold    : fraud probability threshold

    Returns
    -------
    List of suspicious clusters (each cluster is a list of node names)
    """
    try:
        from community import best_partition  # python-louvain
        partition = best_partition(G)
        clusters = {}
        for node, comm_id in partition.items():
            clusters.setdefault(comm_id, []).append(node)

        # Flag clusters with majority high-risk transactions
        suspicious = []
        for comm_id, nodes in clusters.items():
            tx_nodes = [n for n in nodes if n.startswith('TX_')]
            if not tx_nodes:
                continue
            indices = [int(n.split('_')[1]) for n in tx_nodes
                      if int(n.split('_')[1]) < len(fraud_probs)]
            if not indices:
                continue
            mean_prob = np.mean(fraud_probs[indices])
            if mean_prob > threshold:
                suspicious.append(nodes)

        return sorted(suspicious, key=len, reverse=True)

    except ImportError:
        # Fallback: connected components
        components = list(nx.connected_components(G))
        suspicious = []
        for comp in components:
            tx_nodes = [n for n in comp if n.startswith('TX_')]
            if not tx_nodes:
                continue
            indices = [int(n.split('_')[1]) for n in tx_nodes
                      if int(n.split('_')[1]) < len(fraud_probs)]
            if indices and np.mean(fraud_probs[indices]) > threshold:
                suspicious.append(list(comp))
        return suspicious
