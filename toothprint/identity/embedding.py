"""Learned 3D dental-identity embedding — the open gap in the literature.

A DGCNN (Wang et al. 2019, EdgeConv) point-cloud encoder maps an arch to an L2-normalised
descriptor, trained with **sub-centre ArcFace** (Deng et al. 2019/2020) so genuine re-scans
of one arch embed together and different people repel — a *metric*, not a per-pair rigid fit.
Unlike the classical GICP matcher this is permutation-invariant and pooled, so a partial arch
(missing teeth) still embeds near the whole, which is exactly where the rigid method collapses.

Optional: needs ``torch``. Not imported by ``toothprint.identity.__init__`` — the certification
core stays dependency-light. Train/eval drivers live in ``evaluation/scripts``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_idx(x: torch.Tensor, k: int) -> torch.Tensor:
    """k nearest neighbours per point. x: (B, C, N) -> idx: (B, N, k)."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    neg_dist = -xx - inner - xx.transpose(2, 1)
    return neg_dist.topk(k=k, dim=-1)[1]


def edge_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    """EdgeConv graph feature [x_j - x_i, x_i]. x: (B, C, N) -> (B, 2C, N, k)."""
    B, C, N = x.shape
    idx = knn_idx(x, k) + (torch.arange(B, device=x.device).view(-1, 1, 1) * N)
    flat = x.transpose(2, 1).reshape(B * N, C)
    neigh = flat[idx.view(-1)].view(B, N, k, C)
    centre = x.transpose(2, 1).view(B, N, 1, C).repeat(1, 1, k, 1)
    return torch.cat([neigh - centre, centre], dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNN(nn.Module):
    """EdgeConv encoder: (B, N, 3) point cloud -> (B, emb_dim) unit embedding."""

    def __init__(self, emb_dim: int = 256, k: int = 20):
        super().__init__()
        self.k = k

        def conv(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2),
            )

        self.e1, self.e2, self.e3 = conv(6, 64), conv(64 * 2, 64), conv(64 * 2, 128)
        self.fuse = nn.Sequential(
            nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2)
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, emb_dim),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        x = pts.transpose(2, 1)  # (B, 3, N)
        x1 = self.e1(edge_feature(x, self.k)).max(dim=-1)[0]
        x2 = self.e2(edge_feature(x1, self.k)).max(dim=-1)[0]
        x3 = self.e3(edge_feature(x2, self.k)).max(dim=-1)[0]
        g = self.fuse(torch.cat([x1, x2, x3], dim=1))  # (B, 512, N)
        g = torch.cat([g.max(dim=-1)[0], g.mean(dim=-1)], dim=1)  # (B, 1024)
        return F.normalize(self.head(g), dim=1)


class CorrNet(nn.Module):
    """DGCNN-backbone per-point descriptors for learned partial->whole correspondence.

    The global embedding (DGCNN) collapses a whole arch to one vector; that is robust but
    discards the point structure a *partial* query needs to register. CorrNet instead emits a
    unit descriptor per point so a half-arch query can be matched point-to-point against a full
    gallery arch (mutual-NN -> Procrustes), the GeoTransformer-class fix for the rigid PCA-init
    that collapses under tooth loss. Trained with a correspondence (InfoNCE) loss on crop pairs,
    whose point indices give ground-truth matches for free.
    """

    def __init__(self, desc_dim: int = 64, k: int = 20):
        super().__init__()
        self.k = k

        def conv(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2),
            )

        self.e1, self.e2, self.e3 = conv(6, 64), conv(64 * 2, 64), conv(64 * 2, 128)
        self.fuse = nn.Sequential(
            nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2)
        )
        self.desc = nn.Conv1d(512, desc_dim, 1)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        x = pts.transpose(2, 1)  # (B, 3, N)
        x1 = self.e1(edge_feature(x, self.k)).max(dim=-1)[0]
        x2 = self.e2(edge_feature(x1, self.k)).max(dim=-1)[0]
        x3 = self.e3(edge_feature(x2, self.k)).max(dim=-1)[0]
        g = self.fuse(torch.cat([x1, x2, x3], dim=1))  # (B, 512, N)
        return F.normalize(
            self.desc(g), dim=1
        )  # (B, desc_dim, N) unit per-point descriptors


class SubCenterArcFace(nn.Module):
    """Sub-centre ArcFace head — K centres per class absorb intra-arch variation."""

    def __init__(
        self,
        emb_dim: int,
        n_classes: int,
        sub: int = 3,
        s: float = 30.0,
        m: float = 0.4,
    ):
        super().__init__()
        self.n, self.sub, self.s, self.m = n_classes, sub, s, m
        self.W = nn.Parameter(F.normalize(torch.randn(n_classes * sub, emb_dim), dim=1))

    def cosine(self, emb: torch.Tensor) -> torch.Tensor:
        """Un-margined sub-centre cosine to each class — use this for accuracy/retrieval."""
        return (
            (emb @ F.normalize(self.W, dim=1).t())
            .view(-1, self.n, self.sub)
            .max(dim=2)[0]
        )

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = self.cosine(emb).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos)
        margined = torch.cos(theta + self.m * F.one_hot(labels, self.n).float())
        return self.s * margined
