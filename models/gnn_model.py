"""
models/gnn_model.py
--------------------
Heterogeneous Graph Attention Network (HeteroGAT) for fraud detection.

Architecture Choice: GAT > GraphSAGE for fraud because:
- Attention weights are interpretable (which neighbor matters most?)
- Fraudulent nodes have DISTINCT neighbor patterns → attention amplifies this
- Multi-head attention reduces variance from noisy edges

Model:
  Input → HeteroConv(GAT) → BN → ReLU → Dropout
        → HeteroConv(GAT) → BN → ReLU → Dropout
        → HeteroConv(GAT) → BN → ReLU → Dropout
        → Linear → Sigmoid (binary output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GATConv, SAGEConv, HeteroConv,
                                 BatchNorm, Linear, to_hetero)
from torch_geometric.data import HeteroData
from typing import Dict


class FraudGNN(nn.Module):
    """
    Heterogeneous GAT for fraud detection.

    Parameters
    ----------
    metadata       : (node_types, edge_types) from HeteroData
    in_channels    : input feature dimension (for transaction nodes)
    hidden_channels: hidden dimension
    out_channels   : 2 (binary classification)
    num_layers     : number of graph conv layers (2–3 recommended)
    heads          : GAT attention heads
    dropout        : dropout probability
    """

    def __init__(self,
                 metadata,
                 in_channels: int,
                 hidden_channels: int = 128,
                 out_channels: int = 2,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        # Project all node types to hidden_channels
        self.node_projections = nn.ModuleDict()
        for node_type in metadata[0]:
            # We don't know in_channels per type, use lazy init or uniform
            self.node_projections[node_type] = nn.LazyLinear(hidden_channels)

        # Graph Attention Layers (HeteroConv wraps per edge-type convs)
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for layer in range(num_layers):
            is_last = (layer == num_layers - 1)
            out_dim = out_channels if is_last else hidden_channels

            conv_dict = {}
            for edge_type in metadata[1]:
                # Use concat=False to keep dimension fixed after attention
                conv_dict[edge_type] = GATConv(
                    in_channels  = hidden_channels if layer > 0 else hidden_channels,
                    # With concat=False, output dim equals out_channels (not out_channels * heads)
                    out_channels = out_dim,
                    heads        = 1 if is_last else heads,
                    concat       = False,
                    dropout      = dropout,
                    add_self_loops = False
                )

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
            if not is_last:
                self.bns.append(nn.ModuleDict({
                    nt: nn.BatchNorm1d(hidden_channels)
                    for nt in metadata[0]
                }))

        # Final classifier head (operates on transaction node only)
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x_dict: Dict, edge_index_dict: Dict) -> torch.Tensor:
        """
        Forward pass.

        Returns logits for transaction nodes only (shape: [N_tx, 2]).
        """
        # Project each node type to hidden_channels
        h = {nt: F.relu(self.node_projections[nt](x))
             for nt, x in x_dict.items()}

        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict)
            is_last = (i == self.num_layers - 1)
            if not is_last:
                # BN + ReLU + Dropout
                h_new = {
                    nt: F.dropout(
                        F.relu(self.bns[i][nt](feat)),
                        p=self.dropout,
                        training=self.training
                    )
                    for nt, feat in h_new.items()
                    if nt in self.bns[i]
                }
                # Keep node types not updated in this layer by carrying over previous state.
                for nt, feat in h.items():
                    if nt not in h_new:
                        h_new[nt] = feat
            h = h_new

        # Classify transaction nodes only
        tx_emb = h['transaction']
        out = self.classifier(tx_emb)
        return out

    def get_embeddings(self, x_dict: Dict, edge_index_dict: Dict) -> torch.Tensor:
        """Return penultimate-layer embeddings for transaction nodes (for visualization)."""
        h = {nt: F.relu(self.node_projections[nt](x))
             for nt, x in x_dict.items()}
        for i, conv in enumerate(self.convs[:-1]):
            h_new = conv(h, edge_index_dict)
            h_new = {
                nt: F.dropout(F.relu(self.bns[i][nt](feat)),
                               p=self.dropout, training=False)
                for nt, feat in h_new.items()
                if nt in self.bns[i]
            }
            for nt, feat in h.items():
                if nt not in h_new:
                    h_new[nt] = feat
            h = h_new
        return h['transaction']


class FraudGraphSAGE(nn.Module):
    """
    Alternative: Heterogeneous GraphSAGE model.
    Slightly faster training but less interpretable than GAT.
    Use when graph is very large (>1M edges).
    """

    def __init__(self, metadata, in_channels: int,
                 hidden_channels: int = 128, num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        self.node_projections = nn.ModuleDict({
            nt: nn.LazyLinear(hidden_channels)
            for nt in metadata[0]
        })

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for layer in range(num_layers):
            conv_dict = {
                et: SAGEConv(hidden_channels, hidden_channels, normalize=True)
                for et in metadata[1]
            }
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
            self.bns.append(nn.ModuleDict({
                nt: nn.BatchNorm1d(hidden_channels)
                for nt in metadata[0]
            }))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x_dict, edge_index_dict):
        h = {nt: F.relu(self.node_projections[nt](x))
             for nt, x in x_dict.items()}

        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict)
            h = {
                nt: F.dropout(F.relu(self.bns[i][nt](feat)),
                               p=self.dropout, training=self.training)
                for nt, feat in h_new.items()
                if nt in self.bns[i]
            }

        return self.classifier(h['transaction'])
