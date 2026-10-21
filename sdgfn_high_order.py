"""High-order brain-network encoder for SDGFN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob=0.0, training=False):
    """Apply sample-wise stochastic depth to a batched node tensor."""
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    mask = torch.rand((x.shape[0], 1, 1), device=x.device) + keep_prob
    return x / keep_prob * mask.floor()


class HighOrderAttentionBlock(nn.Module):
    """Q-K-V attention modulated by positive and negative edge weights."""

    def __init__(self, dim, dropout=0.3):
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.w_pos = nn.Parameter(torch.tensor(0.1))
        self.w_neg = nn.Parameter(torch.tensor(-0.05))

        # Retained for compatibility with existing state dictionaries.
        self.bias_high = nn.Parameter(torch.zeros(1))

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        _, num_nodes, _ = x.shape
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        sim = torch.matmul(query, key.transpose(-2, -1)) / 0.2

        pos = torch.clamp(adj, 0, 1)
        neg = torch.clamp(adj, -1, 0)
        sim = sim + pos * self.w_pos + neg * self.w_neg

        sim = (sim + sim.transpose(-2, -1)) / 2
        sim = sim + torch.eye(num_nodes, device=sim.device) * 0.1

        mask = adj.abs() > 0.01
        sim = sim.masked_fill(~mask, -1e9)

        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value)
        out = drop_path(out, 0.1, self.training)

        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class HighOrderReadout(nn.Module):
    """Concatenate mean pooling and a sigmoid-weighted node sum."""

    def __init__(self, dim):
        super().__init__()
        self.att = nn.Linear(dim, 1)

    def forward(self, x):
        mean_feat = x.mean(dim=1)
        att_w = torch.sigmoid(self.att(x))
        att_feat = (x * att_w).sum(dim=1)
        return torch.cat([mean_feat, att_feat], dim=1)


class SDGFNHighOrderBranch(nn.Module):
    """Encode high-order networks into graph-level outputs."""

    def __init__(
        self,
        ROInum=246,
        hidden_dim=128,
        layers=3,
        num_classes=128,
        dropout=0.3,
    ):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(ROInum, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                HighOrderAttentionBlock(hidden_dim, dropout)
                for _ in range(layers)
            ]
        )
        self.readout = HighOrderReadout(hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, adj):
        x = self.input(x)

        for block in self.blocks:
            x = block(x, adj)

        features = self.readout(x)
        return self.proj(features)