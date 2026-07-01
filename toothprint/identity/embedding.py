"""Learned 3D dental-identity embedding — the open gap in the literature.

Two interchangeable point-cloud backbones map an arch to an L2-normalised descriptor, trained
with **sub-centre ArcFace** (Deng et al. 2019/2020) so genuine re-scans of one arch embed
together and different people repel — a *metric*, not a per-pair rigid fit. Both are
permutation-invariant and pooled, so a partial arch (missing teeth) still embeds near the whole,
which is exactly where the rigid method collapses.

Backbones (select with :func:`build_embedding_backbone`):

* ``"dgcnn"`` — DGCNN (Wang et al. 2019, EdgeConv). Trained from scratch, no external weights;
  the default, dependency-light (``torch`` only).
* ``"sonata"`` — Point Transformer V3 (Wu et al. 2024) pretrained by **Sonata** self-supervised
  learning (Wu et al., *CVPR 2025*, arXiv 2503.16429; code/weights in Pointcept,
  https://github.com/Pointcept/Pointcept, weights ``facebook/sonata`` on HuggingFace). This is
  the first application of a point-cloud **foundation model** to dental identity — the flagged
  upgrade to the from-scratch DGCNN. Needs ``pointcept``/``sonata`` + weights (lazy-imported,
  see :class:`SonataEmbedding` and ``evaluation/scripts/RUN.md``).

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


class SonataEmbedding(nn.Module):
    """PTv3 encoder pretrained by **Sonata** SSL, wrapped to the same descriptor interface as
    :class:`DGCNN` — ``(B, N, 3) -> (B, emb_dim)`` L2-normalised.

    Sonata (Wu et al., CVPR 2025, arXiv 2503.16429) is a self-supervised *point-cloud foundation
    model*: a Point Transformer V3 encoder pretrained on large indoor/outdoor scans. Nobody has
    applied a point-cloud foundation model to dental identity, so this is first-mover work — we
    reuse the frozen (or fine-tuned) Sonata features and put the *same* sub-centre ArcFace head +
    L2-norm descriptor on top, keeping the crop-hardened training recipe.

    Dependencies (``pointcept`` / ``sonata`` + weights) are **lazy-imported**: the class can be
    constructed and introspected without them, and only ``.load()`` / ``forward`` touch Pointcept,
    raising a clear :class:`ImportError` with install instructions if it is missing. This keeps the
    package importable on a CPU box while making the wrapper, config, and training script real.

    Parameters
    ----------
    emb_dim : final descriptor dimension (matches DGCNN's 256 by default).
    feat_dim : PTv3 encoder output channels feeding the projection head (Sonata's default is 512;
        overridden automatically from the loaded model when possible).
    repo_id : HuggingFace repo for the pretrained weights (default ``"facebook/sonata"``).
    model_name : Sonata model tag passed to ``sonata.load`` (default ``"sonata"``).
    grid_size : voxelisation grid used to build the PTv3 input (mm-normalised arch is unit-scaled,
        so the default 0.02 gives ~a few-thousand voxels; document/tune in RUN.md).
    freeze_backbone : if True (default) the pretrained encoder is frozen and only the projection
        head trains — the recommended low-data recipe for 150 subjects.
    """

    _INSTALL_HINT = (
        "Sonata/PTv3 backbone requires Pointcept. Install with:\n"
        "  pip install spconv-cu120 torch-scatter\n"
        "  pip install 'git+https://github.com/Pointcept/Pointcept.git#subdirectory=libs/pointops'\n"
        "  pip install sonata-3d   # or: pip install 'git+https://github.com/Pointcept/Pointcept.git'\n"
        "Pretrained weights are fetched automatically from HuggingFace 'facebook/sonata'.\n"
        "See evaluation/scripts/RUN.md for the exact, tested command sequence."
    )

    def __init__(
        self,
        emb_dim: int = 256,
        feat_dim: int = 512,
        repo_id: str = "facebook/sonata",
        model_name: str = "sonata",
        grid_size: float = 0.02,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.repo_id = repo_id
        self.model_name = model_name
        self.grid_size = grid_size
        self.freeze_backbone = freeze_backbone
        self.backbone = None  # lazily populated by .load()
        # Projection head: pooled PTv3 features -> unit descriptor. Real params so the head is
        # trainable/checkpointable even before the (heavy) backbone weights are fetched.
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, emb_dim),
        )

    def load(self):
        """Fetch + build the Sonata-pretrained PTv3 encoder. Idempotent. Raises ImportError with
        an install hint if Pointcept/Sonata is unavailable."""
        if self.backbone is not None:
            return self
        try:
            import sonata  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised only without the dep
            raise ImportError(self._INSTALL_HINT) from e
        model = sonata.load(self.model_name, repo_id=self.repo_id)
        if self.freeze_backbone:
            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
        # Sync the projection input dim to the real encoder width when discoverable.
        enc_dim = getattr(model, "enc_channels", None)
        if isinstance(enc_dim, (list, tuple)) and enc_dim:
            enc_dim = enc_dim[-1]
        if isinstance(enc_dim, int) and enc_dim != self.feat_dim:
            self.feat_dim = enc_dim
            self.head[0] = nn.Linear(enc_dim, 512, bias=False)
        self.backbone = model
        return self

    def _to_point_dict(self, pts: torch.Tensor):
        """Turn a batched ``(B, N, 3)`` cloud into Pointcept's Point dict (coord/feat/batch/grid).
        Coordinates double as features (no colour on a bare arch); documented in RUN.md."""
        B, N, _ = pts.shape
        coord = pts.reshape(B * N, 3)
        batch = torch.arange(B, device=pts.device).repeat_interleave(N)
        return {
            "coord": coord,
            "feat": coord,
            "batch": batch,
            "grid_size": self.grid_size,
        }

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            self.load()
        point = self.backbone(self._to_point_dict(pts))
        # Pointcept returns a Point object with .feat (per-point) and .batch; mean+max pool per
        # cloud to a global descriptor, mirroring DGCNN's pooled head.
        feat, batch = point["feat"], point["batch"]
        B = int(batch.max().item()) + 1
        pooled = []
        for b in range(B):
            f = feat[batch == b]
            pooled.append(torch.cat([f.max(0)[0], f.mean(0)], dim=0))
        g = torch.stack(pooled, dim=0)
        # head expects feat_dim; pooled is 2*feat_dim (max||mean) -> adapt head input on first use.
        if self.head[0].in_features != g.shape[1]:
            self.head[0] = nn.Linear(g.shape[1], 512, bias=False).to(g.device)
        return F.normalize(self.head(g), dim=1)


def build_embedding_backbone(backbone: str = "dgcnn", emb_dim: int = 256, **kwargs) -> nn.Module:
    """Factory selecting the identity embedding backbone.

    Parameters
    ----------
    backbone : ``"dgcnn"`` (default, from-scratch EdgeConv) or ``"sonata"`` (PTv3 foundation model).
    emb_dim : descriptor dimension shared by both backbones and the ArcFace head.
    **kwargs : forwarded to the chosen module (e.g. ``k`` for DGCNN; ``repo_id``, ``grid_size``,
        ``freeze_backbone`` for Sonata).

    Both returned modules share the contract ``forward((B, N, 3)) -> (B, emb_dim)`` L2-normalised,
    so :class:`SubCenterArcFace` and every training/eval driver are backbone-agnostic.
    """
    b = backbone.lower()
    if b == "dgcnn":
        return DGCNN(emb_dim=emb_dim, **kwargs)
    if b == "sonata":
        return SonataEmbedding(emb_dim=emb_dim, **kwargs)
    raise ValueError(f"unknown backbone {backbone!r}; choose 'dgcnn' or 'sonata'")


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
