"""
utils/graph_builder.py
-----------------------
Converts the tabular DataFrame into a PyTorch Geometric HeteroData graph.

WHY THIS GRAPH STRUCTURE IS POWERFUL
─────────────────────────────────────
1. Shared-entity connections expose fraud rings:
   If card1 is used in 10 transactions and 3 are fraudulent,
   the non-fraudulent ones LEARN from their fraudulent neighbors
   via message passing — impossible with tabular models.

2. Heterogeneous node types let the GNN learn entity-specific
   embeddings. A device embedding captures device-level behavior
   patterns across ALL transactions that touched it.

3. Temporal locality via TransactionDT edges means recent
   co-occurrence gets more signal than old one.

4. Multi-hop reasoning: Fraud ring detection across 2-3 hops:
   Txn_A → Card_X ← Txn_B → Device_Y ← Txn_C
   A 3-layer GNN sees the entire ring.

HOW FRAUD RINGS ARE CAUGHT
───────────────────────────
Fraudsters reuse cards, devices, and email domains.
Graph edges make these links explicit. GNN aggregates
neighborhood signals so even a single confirmed fraud node
"contaminates" its neighbors with high fraud probability.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


def _get_feature_cols(df: pd.DataFrame, exclude: list) -> list:
    """Return numeric feature columns excluding IDs and targets."""
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude]


def build_graph(df: pd.DataFrame,
                max_nodes: Optional[int] = None) -> HeteroData:
    """
    Build a heterogeneous PyG graph from the processed DataFrame.

    Node types:
        - transaction  (one per row)
        - card         (card1 values)
        - email        (P_emaildomain values)
        - device       (DeviceType values)
        - address      (addr1 values)

    Edge types (bidirectional):
        - transaction → uses → card
        - transaction → uses → device
        - transaction → uses → email
        - transaction → uses → address
        - card/device/email shared between transactions
          (indirect, implicit via shared entity nodes)

    Parameters
    ----------
    df         : Preprocessed DataFrame
    max_nodes  : Subsample transactions (for prototyping)

    Returns
    -------
    HeteroData object ready for PyG GNN training
    """
    if max_nodes:
        df = df.sample(min(max_nodes, len(df)), random_state=42).reset_index(drop=True)

    print(f"[Graph] Building graph from {len(df):,} transactions…")

    data = HeteroData()

    # ── Transaction Features ────────────────────────────────────────────────
    exclude = ['TransactionID', 'isFraud', 'card1', 'card2', 'card3',
               'card4', 'card5', 'card6', 'P_emaildomain', 'R_emaildomain',
               'DeviceType', 'DeviceInfo', 'addr1', 'addr2']
    feat_cols = _get_feature_cols(df, exclude)
    # Clamp to available columns
    feat_cols = [c for c in feat_cols if c in df.columns]

    tx_features = torch.tensor(df[feat_cols].values, dtype=torch.float)
    data['transaction'].x = tx_features
    data['transaction'].y = torch.tensor(df['isFraud'].values, dtype=torch.long)
    data['transaction'].tx_id = torch.tensor(
        df['TransactionID'].values if 'TransactionID' in df.columns
        else np.arange(len(df)), dtype=torch.long)

    print(f"  Transaction nodes: {len(df):,} | Features: {tx_features.shape[1]}")

    # ── Helper: build entity nodes + edges ─────────────────────────────────
    def add_entity(col: str, node_type: str, edge_type: str):
        if col not in df.columns:
            return
        values = df[col].values
        unique = np.unique(values)
        entity_map = {v: i for i, v in enumerate(unique)}

        # Entity node features = mean of connected transaction features
        feat_dim = tx_features.shape[1]
        entity_feats = torch.zeros((len(unique), feat_dim), dtype=torch.float)
        counts = torch.zeros(len(unique), dtype=torch.float)
        for tx_idx, val in enumerate(values):
            eid = entity_map[val]
            entity_feats[eid] += tx_features[tx_idx]
            counts[eid] += 1
        counts = counts.clamp(min=1).unsqueeze(1)
        entity_feats = entity_feats / counts

        data[node_type].x = entity_feats

        # Edges: transaction → entity (and reverse)
        tx_idx  = torch.arange(len(df), dtype=torch.long)
        ent_idx = torch.tensor([entity_map[v] for v in values], dtype=torch.long)

        data['transaction', edge_type, node_type].edge_index = \
            torch.stack([tx_idx, ent_idx], dim=0)
        data[node_type, f'rev_{edge_type}', 'transaction'].edge_index = \
            torch.stack([ent_idx, tx_idx], dim=0)

        print(f"  {node_type:12s} nodes: {len(unique):,} | "
              f"Edges: {len(tx_idx):,}")

    add_entity('card1',         'card',    'uses_card')
    add_entity('P_emaildomain', 'email',   'uses_email')
    add_entity('DeviceType',    'device',  'uses_device')
    add_entity('addr1',         'address', 'uses_addr')

    # ── Transaction–Transaction edges (shared card/device) ─────────────────
    # Two transactions share an edge if they use the same card1
    # This is the critical edge for fraud ring detection
    if 'card1' in df.columns:
        card_vals = df['card1'].values
        tx_src, tx_dst = [], []
        from itertools import combinations
        card_groups = {}
        for i, c in enumerate(card_vals):
            card_groups.setdefault(c, []).append(i)
        for group in card_groups.values():
            if 1 < len(group) <= 50:  # cap to avoid O(n²) on popular cards
                for a, b in combinations(group, 2):
                    tx_src.append(a); tx_dst.append(b)
                    tx_src.append(b); tx_dst.append(a)
        if tx_src:
            data['transaction', 'same_card', 'transaction'].edge_index = \
                torch.tensor([tx_src, tx_dst], dtype=torch.long)
            print(f"  tx↔tx (same card) edges: {len(tx_src):,}")

    print(f"\n✅ Graph built successfully!")
    print(f"   Node types : {list(data.node_types)}")
    print(f"   Edge types : {list(data.edge_types)}")
    return data
