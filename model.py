#!/usr/bin/env python3
"""
model.py — GAT-based surrogate model for the PC-MT-EVRP-RT-D objective value.

Architecture (all components controlled by constructor arguments):

    1. Node encoder      : 2-layer MLP  in_channels -> hidden_dim -> hidden_dim
    2. Edge encoder      : 2-layer MLP  edge_in_channels -> edge_hidden_dim -> edge_hidden_dim
    3. GAT stack (K layers), each:
           h -> GATv2Conv(edge_dim=edge_hidden_dim) -> GraphNorm -> ReLU -> Dropout
                                                                          \
                                                                           + residual
    4. Sum pooling                  : [N, hidden_dim] -> [B, hidden_dim]
    5. (optional) Global conditioning: concat [battery_capacity, vehicle_load_capacity]
    6. Regression head MLP          : (hidden_dim + #global_feats) -> ... -> 1

Defaults match the agreed Phase B spec; everything is overridable to enable
hyperparameter sweeps.

Run the file directly to see a structural smoke test:

    python model.py
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch, Data
#from torch_geometric.nn import GATv2Conv, GraphNorm, global_add_pool
from torch_geometric.nn import GATv2Conv, GraphNorm, global_mean_pool

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GATSurrogate(nn.Module):
    """
    Graph Attention Network for graph-level regression on EVRP objective values.

    Args:
        in_channels:             dimension of input node features. With Phase A's
                                 default pe_dim=16 this is 2*16 + 4 = 36.
        edge_in_channels:        dimension of input edge features (default 2:
                                 [distance, distance / battery_capacity]).
        hidden_dim:              node-embedding width through the GAT stack.
                                 Must be divisible by `heads` when concat_heads=True.
        edge_hidden_dim:         edge-embedding width after the edge encoder.
        num_gat_layers:          number of stacked GATv2Conv layers.
        heads:                   number of attention heads per GAT layer.
        concat_heads:            True  -> heads are concatenated (output per layer
                                          has width hidden_dim, achieved by setting
                                          GATv2Conv out_channels = hidden_dim // heads)
                                 False -> heads are averaged (out_channels = hidden_dim)
        dropout:                 dropout applied inside the GAT body
                                 (both on attention weights and after activations).
        head_dropout:            dropout used between layers of the regression head.
        head_depth:              number of linear layers in the regression head
                                 (>=1; 2 is the recommended default).
        use_residual:            add the input of each GAT block back to its output.
        use_global_conditioning: concatenate [battery_capacity, vehicle_load_capacity]
                                 onto the pooled graph embedding before the head.
        num_global_features:     number of scalar graph-level features being
                                 concatenated when use_global_conditioning=True.
    """

    def __init__(
        self,
        in_channels: int = 36,
        edge_in_channels: int = 2,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 32,
        num_gat_layers: int = 4,
        heads: int = 4,
        concat_heads: bool = True,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        head_depth: int = 2,
        use_residual: bool = True,
        use_global_conditioning: bool = True,
        num_global_features: int = 2,
    ):
        super().__init__()

        if concat_heads and hidden_dim % heads != 0:
            raise ValueError(
                f"concat_heads=True requires hidden_dim % heads == 0; "
                f"got hidden_dim={hidden_dim}, heads={heads}"
            )
        if head_depth < 1:
            raise ValueError(f"head_depth must be >= 1, got {head_depth}")

        self.in_channels             = in_channels
        self.edge_in_channels        = edge_in_channels
        self.hidden_dim              = hidden_dim
        self.edge_hidden_dim         = edge_hidden_dim
        self.num_gat_layers          = num_gat_layers
        self.heads                   = heads
        self.concat_heads            = concat_heads
        self.dropout_p               = dropout
        self.head_dropout_p          = head_dropout
        self.head_depth              = head_depth
        self.use_residual            = use_residual
        self.use_global_conditioning = use_global_conditioning
        self.num_global_features     = num_global_features

        # ---- Node encoder: 36 -> hidden -> hidden ----
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- Edge encoder: 2 -> edge_hidden -> edge_hidden ----
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_channels, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
        )

        # ---- GAT stack ----
        # In concat mode, GATv2's output dim is out_channels * heads; pick
        # out_channels so that the post-layer width equals hidden_dim and
        # residual addition needs no projection.
        out_channels_per_layer = hidden_dim // heads if concat_heads else hidden_dim
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=out_channels_per_layer,
                heads=heads,
                concat=concat_heads,
                edge_dim=edge_hidden_dim,
                dropout=dropout,
                add_self_loops=False,   # we already have a complete graph
            )
            for _ in range(num_gat_layers)
        ])
        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_gat_layers)])

        # ---- Regression head ----
        head_input_dim = hidden_dim + (num_global_features if use_global_conditioning else 0)

        head_layers = []
        if head_depth == 1:
            head_layers.append(nn.Linear(head_input_dim, 1))
        else:
            # Taper: H+G -> H/2 -> H/4 -> ... -> 1
            current_dim = head_input_dim
            next_dim = hidden_dim // 2
            for i in range(head_depth - 1):
                head_layers.append(nn.Linear(current_dim, next_dim))
                head_layers.append(nn.ReLU())
                head_layers.append(nn.Dropout(head_dropout))
                current_dim = next_dim
                next_dim = max(current_dim // 2, 1)
            head_layers.append(nn.Linear(current_dim, 1))
        self.regression_head = nn.Sequential(*head_layers)

    # ---- forward pass ------------------------------------------------------
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        batch: a `torch_geometric.data.Batch` produced by a DataLoader.
               Expected attributes: x, edge_index, edge_attr, batch.
               If use_global_conditioning=True, also: battery_capacity,
               vehicle_load_capacity (each shape [num_graphs] post-batching).
        Returns: tensor of shape [num_graphs, 1].
        """
        x          = batch.x                # [N, in_channels]
        edge_index = batch.edge_index       # [2, E]
        edge_attr  = batch.edge_attr        # [E, edge_in_channels]
        batch_idx  = batch.batch            # [N]

        h = self.node_encoder(x)            # [N, hidden_dim]
        e = self.edge_encoder(edge_attr)    # [E, edge_hidden_dim]

        for gat, norm in zip(self.gat_layers, self.norms):
            h_in = h
            h = gat(h, edge_index, edge_attr=e)
            h = norm(h, batch_idx)
            if self.use_residual:
                h = h + h_in
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)

        g = global_mean_pool(h, batch_idx)                   # [B, hidden_dim]

        if self.use_global_conditioning:
            bat = batch.battery_capacity.view(-1, 1)        # [B, 1]
            vlc = batch.vehicle_load_capacity.view(-1, 1)   # [B, 1]
            g = torch.cat([g, bat, vlc], dim=-1)            # [B, hidden_dim + 2]

        return self.regression_head(g)                       # [B, 1]

    # ---- diagnostics -------------------------------------------------------
    def num_parameters(self) -> Dict[str, int]:
        """Return a breakdown of parameter counts by component."""
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        return {
            "node_encoder":    _count(self.node_encoder),
            "edge_encoder":    _count(self.edge_encoder),
            "gat_layers":      _count(self.gat_layers),
            "graph_norms":     _count(self.norms),
            "regression_head": _count(self.regression_head),
            "total":           _count(self),
        }


# ---------------------------------------------------------------------------
# Smoke test (run `python model.py`)
# ---------------------------------------------------------------------------
def _make_dummy_graph(num_nodes: int = 10) -> Data:
    """Synthetic Data object matching the Phase A output layout."""
    x = torch.randn(num_nodes, 36)
    # Complete graph (no self-loops)
    row, col = torch.meshgrid(
        torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
    )
    mask = row != col
    edge_index = torch.stack([row[mask], col[mask]], dim=0)
    edge_attr = torch.randn(edge_index.size(1), 2)

    d = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.randn(1),
    )
    d.battery_capacity      = torch.tensor([1.0])
    d.vehicle_load_capacity = torch.tensor([1.0])
    return d


def _smoke_test() -> None:
    graphs = [_make_dummy_graph(n) for n in (8, 12, 16)]
    batch = Batch.from_data_list(graphs)

    model = GATSurrogate()
    print(model)
    print()
    print("Parameter counts:")
    for k, v in model.num_parameters().items():
        print(f"  {k:18s} = {v:>10,}")

    model.eval()
    with torch.no_grad():
        pred = model(batch)
    print(f"\nInput:  {len(graphs)} graphs, batch.x.shape = {tuple(batch.x.shape)}")
    print(f"Output: shape={tuple(pred.shape)}  (expected [{len(graphs)}, 1])")
    print(f"        values={pred.squeeze().tolist()}")


if __name__ == "__main__":
    _smoke_test()
