"""Low-order brain-network encoder for SDGFN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2(nn.Module):
    """GATv2-style local attention with adjacency weighting."""

    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        self.edge_weight = nn.Parameter(torch.tensor(0.05))
        self.residual = (
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        _, num_nodes, _ = x.shape
        h = self.fc(x)

        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        a_input = torch.cat([h_i, h_j], dim=-1)

        scores = self.attn(a_input).squeeze(-1)
        scores = scores + torch.tanh(adj * self.edge_weight)

        adj_mask = adj > 0
        scores = scores.masked_fill(~adj_mask, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = attn * adj
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = self.dropout(attn)

        out = torch.matmul(attn, h)
        out = self.norm(out + self.residual(x))
        return self.act(out)


class GraphGPSLayer(nn.Module):
    """Global attention with adjacency propagation and a feed-forward block."""

    def __init__(self, dim, heads=4, dropout=0.3):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            dim,
            heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, adj):
        x = x + self.alpha * torch.matmul(adj, x)

        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class FusionBlock(nn.Module):
    """Feature-wise gated fusion of local and global representations."""

    def __init__(self, dim, dropout=0.3):
        super().__init__()

        self.gat = GATv2(dim, dim, dropout)
        self.gps = GraphGPSLayer(dim, 4, dropout)
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        g1 = self.gat(x, adj)
        g2 = self.gps(x, adj)

        gate = self.gate(torch.cat([g1, g2, x], dim=-1))
        h = gate * g1 + (1 - gate) * g2
        h = self.dropout(h)
        return self.norm(h + x)


class SDGFNLowOrderBranch(nn.Module):
    """Encode low-order networks into graph-level outputs."""

    def __init__(
        self,
        ROInum=246,
        hidden_dim=128,
        layers=2,
        num_classes=2,
        dropout=0.3,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(ROInum, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [FusionBlock(hidden_dim, dropout) for _ in range(layers)]
        )
        self.att_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, adj)

        # Preserve the original signed, sum-normalized readout.
        att = self.att_pool(x)
        att = att / (att.sum(dim=1, keepdim=True) + 1e-8)
        x = torch.sum(x * att, dim=1)
        return self.classifier(x)