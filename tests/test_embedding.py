"""Tests for the identity embedding backbone selection API (DGCNN + Sonata/PTv3).

The conformal certification core is untouched; these only cover the learned-embedding wrapper:
backbone factory, descriptor shape/normalisation, and the graceful missing-dependency path for
the Sonata foundation-model backbone.
"""
import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toothprint.identity.embedding import (  # noqa: E402
    DGCNN,
    SonataEmbedding,
    SubCenterArcFace,
    build_embedding_backbone,
)

_HAS_SONATA = importlib.util.find_spec("sonata") is not None


# --- backbone selection API ------------------------------------------------


def test_build_backbone_dgcnn_default():
    net = build_embedding_backbone(emb_dim=128)
    assert isinstance(net, DGCNN)


def test_build_backbone_sonata():
    net = build_embedding_backbone("sonata", emb_dim=128)
    assert isinstance(net, SonataEmbedding)
    assert net.emb_dim == 128


def test_build_backbone_case_insensitive():
    assert isinstance(build_embedding_backbone("DGCNN"), DGCNN)
    assert isinstance(build_embedding_backbone("Sonata"), SonataEmbedding)


def test_build_backbone_unknown_raises():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_embedding_backbone("pointnet")


# --- descriptor shape / normalisation --------------------------------------


def test_dgcnn_descriptor_shape_and_unit_norm():
    net = build_embedding_backbone("dgcnn", emb_dim=64).eval()
    pts = torch.randn(4, 256, 3)
    with torch.no_grad():
        emb = net(pts)
    assert emb.shape == (4, 64)
    norms = emb.norm(dim=1).numpy()
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_arcface_head_consumes_descriptor():
    net = build_embedding_backbone("dgcnn", emb_dim=64).eval()
    head = SubCenterArcFace(64, n_classes=5)
    with torch.no_grad():
        emb = net(torch.randn(3, 256, 3))
        cos = head.cosine(emb)
    assert cos.shape == (3, 5)


def test_sonata_projection_head_is_real_module():
    # head must be a genuine trainable module even before the (heavy) backbone is fetched
    net = SonataEmbedding(emb_dim=64, feat_dim=512)
    assert net.backbone is None
    assert any(p.requires_grad for p in net.head.parameters())


# --- graceful missing-dependency error -------------------------------------


@pytest.mark.skipif(_HAS_SONATA, reason="Pointcept/sonata installed; missing-dep path not exercisable")
def test_sonata_load_missing_dependency_raises_with_hint():
    net = SonataEmbedding(emb_dim=64)
    with pytest.raises(ImportError, match="Pointcept"):
        net.load()


@pytest.mark.skipif(_HAS_SONATA, reason="Pointcept/sonata installed; missing-dep path not exercisable")
def test_sonata_forward_missing_dependency_raises():
    net = SonataEmbedding(emb_dim=64)
    with pytest.raises(ImportError, match="pip install"):
        net(torch.randn(2, 128, 3))
