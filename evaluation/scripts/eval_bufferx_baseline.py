#!/usr/bin/env python3
"""Benchmark **BUFFER-X** zero-shot registration against our CorrNet on the SAME partial-overlap
identity protocol (#1), so the numbers are directly comparable — measured on **real Teeth3DS+**
upper arches.

BUFFER-X (Kim et al., *ICCV 2025*, arXiv 2503.07940; code https://github.com/MIT-SPARK/BUFFER-X)
is a zero-shot point-cloud registration method — a generalist correspondence + pose estimator
tuned for cross-domain transfer, trained on **indoor RGB-D scans (3DMatch)**. Our custom CorrNet
(toothprint.identity.embedding.CorrNet) is a dental-specialised per-point descriptor. This script
runs BUFFER-X on the **identical** protocol as eval_correspondence.py — keep-0.5 / keep-0.3 crops
(planar half-cut + realistic discrete whole-tooth dropout), Rank-1 gallery identification — and
writes bufferx_baseline.json next to correspondence_identity.json for a head-to-head table.

CorrNet reference numbers to beat (realistic whole-tooth dropout, from correspondence_identity.json):

    protocol            CorrNet Rank-1
    ------------------  --------------
    teeth   keep-0.5    ~0.87   (realistic dropout)
    teeth   keep-0.3    ~0.57   (realistic dropout)
    planar  keep-0.5    ~0.91   (clean half-cut, easier)
    planar  keep-0.3    ~0.80

    (baselines for context: crop-hardened embedding ~0.64 @ keep-0.5, rigid GICP ~0.23 @ keep-0.5)

**The honest question this answers:** does a generalist zero-shot registrar trained on indoor
room scans transfer to dental micro-geometry with no retraining? A poor result is a *legitimate*
answer — it quantifies the value of the dental-specialised descriptor.

BUFFER-X is loaded from the built third-party tree under splatreg (`third_party_models/BUFFER-X`,
pretrained 3DMatch Desc/Pose checkpoints + built CUDA neighbour/subsampling ext); if that tree /
its weights / its CUDA extensions are unavailable the script prints the setup and exits without
fabricating numbers. It reuses the crop geometry and Rank-1 scorer from eval_correspondence.py so
the two evaluations differ only in the matcher under test (and BUFFER-X, an indoor-scan method,
gets a denser point budget so it is not handicapped on point count — its own inference is
unchanged).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from train_correspondence import load_norm, rot  # noqa: E402  (load_norm handles any mesh path)
from eval_correspondence import rank1_auc  # noqa: E402  (same Rank-1 / AUC scorer)

OUT = Path(__file__).resolve().parents[1] / "results" / "bufferx_baseline.json"

# Real Teeth3DS+ upper arches (env TP_TEETH3DS -> <id>/<id>*.obj); see paths.py / DATA_GATE.md.
DATA = paths.TEETH3DS
N = int(os.environ.get("TP_BUFFERX_N", "40"))       # arch subset (gallery == probe set)
REPS = int(os.environ.get("TP_BUFFERX_REPS", "1"))  # BUFFER-X is deterministic-ish; 1 rep by default (n^2 regs each)
NP = int(os.environ.get("TP_BUFFERX_NP", "8000"))   # dense sampling: keep-0.3 crops must stay >~2000 pts for BUFFER-X FPS
MODES = os.environ.get("TP_BUFFERX_MODES", "teeth,planar").split(",")
KEEPS = [float(x) for x in os.environ.get("TP_BUFFERX_KEEPS", "0.5,0.3").split(",")]

# BUFFER-X lives in the built splatreg third-party tree by default (see repo facts / RUN.md).
_DEFAULT_REPO = (Path.home() / "workspace/brain/workspace/repos/splatreg/third_party_models/BUFFER-X")
BUFFERX_REPO = Path(os.environ.get("BUFFERX_REPO", str(_DEFAULT_REPO)))

_INSTALL_HINT = (
    "BUFFER-X is not loadable. This script loads the pretrained 3DMatch model from a built\n"
    "BUFFER-X tree (default: splatreg/third_party_models/BUFFER-X). It needs:\n"
    "  - the repo on disk with snapshot/threedmatch/{Desc,Pose}/best.pth,\n"
    "  - its CUDA neighbour/subsampling C++ extensions built (cpp_wrappers/*.so),\n"
    "  - a venv with torch(+CUDA), pointnet2_ops, knn_cuda, kornia, nibabel, open3d.\n"
    "Set BUFFERX_REPO to the repo root. See evaluation/scripts/RUN.md and\n"
    "https://github.com/MIT-SPARK/BUFFER-X . (No numbers are fabricated when it is absent.)\n"
)


def _load_bufferx():
    """Load the pretrained BUFFER-X 3DMatch model from the built splatreg tree and return a
    ``register(src (Ns,3), dst (Nd,3)) -> (R (3,3), t (3,), score float)`` callable, or raise
    ImportError with setup instructions.

    Mirrors the working loader in ``splatreg/splatreg/align_features.py::_load_bufferx``: the
    released Desc/Pose checkpoints are BOTH full-model state_dicts, so they load into the whole
    model (strict=False), Desc first then Pose so the pose-stage weights win. Inference mirrors the
    upstream dataloader for a fresh dataset: sphericity-based voxel-size estimation + voxel
    downsample -> the model's test-mode forward (which does its own multi-scale FPS + density-aware
    radius estimation, so it is scale-adaptive and needs no dental retraining)."""
    import torch

    if not torch.cuda.is_available():
        raise ImportError(_INSTALL_HINT + "\n(CUDA is required for BUFFER-X's pointnet2/knn ops.)")
    pose_w = BUFFERX_REPO / "snapshot" / "threedmatch" / "Pose" / "best.pth"
    desc_w = BUFFERX_REPO / "snapshot" / "threedmatch" / "Desc" / "best.pth"
    ext_ok = any((BUFFERX_REPO / "cpp_wrappers" / "cpp_subsampling").glob("*.so")) and any(
        (BUFFERX_REPO / "cpp_wrappers" / "cpp_neighbors").glob("*.so"))
    if not (pose_w.is_file() and desc_w.is_file() and ext_ok):
        raise ImportError(_INSTALL_HINT + f"\n(looked under BUFFERX_REPO={BUFFERX_REPO})")

    if str(BUFFERX_REPO) not in sys.path:
        sys.path.insert(0, str(BUFFERX_REPO))
    try:
        import open3d as o3d
        from config import make_cfg  # type: ignore
        from models.BUFFERX import BufferX  # type: ignore
        from utils.tools import sphericity_based_voxel_analysis  # type: ignore
    except ImportError as e:  # pragma: no cover - only without the built tree
        raise ImportError(_INSTALL_HINT + f"\n(import error: {e})") from e

    dev = torch.device("cuda")
    cfg = make_cfg("3DMatch", str(BUFFERX_REPO / "snapshot"))
    cfg.stage = "test"
    model = BufferX(cfg).to(dev).eval()
    model.load_state_dict(torch.load(str(desc_w), map_location="cpu"), strict=False)
    model.load_state_dict(torch.load(str(pose_w), map_location="cpu"), strict=False)

    def _to_fds(pts):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(pts, np.float64))
        return pc

    def register(src, dst):
        ps, pt = _to_fds(src), _to_fds(dst)
        vox, _sph, _al = sphericity_based_voxel_analysis(ps, pt)  # upstream dataloader step (scale-adaptive)
        ps = ps.voxel_down_sample(vox)
        pt = pt.voxel_down_sample(vox)
        s = np.asarray(ps.points, np.float32)
        t = np.asarray(pt.points, np.float32)
        data_source = {
            "src_fds_pcd": torch.from_numpy(s).to(dev),
            "tgt_fds_pcd": torch.from_numpy(t).to(dev),
            "is_aligned_to_global_z": False,
        }
        with torch.no_grad():
            out = model(data_source)
        pose = out[0] if isinstance(out, (tuple, list)) else out
        n_inliers = float(out[2]) if isinstance(out, (tuple, list)) and len(out) > 2 else 0.0
        T = np.asarray(pose, np.float64).reshape(4, 4)
        return T[:3, :3], T[:3, 3], n_inliers

    return register


def dense_crop(cloud, rng, keep, mode):
    """Partial query at full point density (mirror of eval_correspondence.crop_query geometry,
    minus its resample-to-M step so BUFFER-X keeps enough points). mode='planar' = clean
    half-space cut (easy); mode='teeth' = REALISTIC discrete whole-tooth dropout."""
    p = cloud @ rot(rng).T
    c = p.mean(0)
    if mode == "planar":
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        proj = (p - c) @ n; p = p[proj >= np.quantile(proj, 1 - keep)]
    else:                                                                    # discrete whole-tooth dropout
        u = np.linalg.svd(p - c, full_matrices=False)[2][0]                  # arch axis
        proj = (p - c) @ u
        band = np.clip(np.digitize(proj, np.linspace(proj.min(), proj.max(), 15)[1:-1]), 0, 13)
        teeth = np.unique(band); nk = max(2, int(round(keep * len(teeth))))
        kept = rng.choice(teeth, nk, replace=False)
        p = p[np.isin(band, kept)]
    return (p + rng.normal(0, 0.01, p.shape)).astype(np.float32)


def residual_bufferx(register, qpts, gtree):
    """Score a query crop against a gallery arch: register, then mean nearest-neighbour residual
    of every aligned query point to the gallery cloud (lower = better match), mirroring
    eval_correspondence.residual's honest all-query-point metric. ``gtree`` is a prebuilt KD-tree
    over the (dense) gallery arch."""
    R, t, _score = register(qpts, gtree.data)
    aligned = qpts @ R.T + t
    d, _ = gtree.query(aligned, k=1)
    return float(np.mean(d))


def main():
    try:
        register = _load_bufferx()
    except ImportError as e:
        print(str(e))
        print("\nAborting: cannot load BUFFER-X here. Wrapper + protocol are wired and ready.")
        sys.exit(1)

    meshes = sorted(Path(DATA).glob("*/*.obj"))[:N]
    if not meshes:
        print(f"No Teeth3DS+ arches under {DATA} (set TP_TEETH3DS). Aborting.")
        sys.exit(1)
    gallery = [g for g in (load_norm(m, NP) for m in meshes) if g is not None]
    qbase = [load_norm(m, NP) for m in meshes]                                # independent sampling for queries
    qbase = [q for q, g in zip(qbase, gallery) if q is not None]
    n = len(gallery)
    gtrees = [cKDTree(g) for g in gallery]
    print(f"bufferx eval (Teeth3DS+, REAL): {n} arches, NP={NP}, REPS={REPS}, "
          f"modes={MODES}, keeps={KEEPS}", flush=True)

    res = {"dataset": "Teeth3DS+ (real intraoral upper arches)",
           "n_arches": n, "reps": REPS, "n_points": NP,
           "matcher": "BUFFER-X (arXiv 2503.07940, ICCV 2025) — zero-shot, pretrained 3DMatch, no dental retraining",
           "protocol": "eval_correspondence.py crop geometry (planar / realistic whole-tooth dropout), Rank-1 gallery ID",
           "corrnet_reference": {"note": "authoritative values in correspondence_identity.json"},
           "results": {}}
    # Pull CorrNet + GICP references from the committed artifact so the table is self-consistent.
    try:
        cr = json.loads((OUT.parent / "correspondence_identity.json").read_text())
        res["corrnet_reference"].update({k: v.get("corrnet_rank1") for k, v in cr["results"].items()})
        res["gicp_reference"] = cr.get("baselines_planar_reference")
    except Exception:
        pass

    t_start = time.time()
    for mode in MODES:
        for keep in KEEPS:
            r1s, aucs = [], []
            for r in range(REPS):
                qpts = [dense_crop(qbase[i], np.random.default_rng(300 + 91 * r + i), keep, mode)
                        for i in range(n)]
                G = np.array([[residual_bufferx(register, qpts[i], gtrees[j]) for j in range(n)]
                              for i in range(n)])
                r1, auc = rank1_auc(G); r1s.append(r1); aucs.append(auc)
            res["results"][f"{mode}_keep{keep}"] = {
                "bufferx_rank1": round(float(np.mean(r1s)), 3),
                "std": round(float(np.std(r1s)), 3),
                "auc": round(float(np.mean(aucs)), 3)}
            print(f"  {mode:6s} keep {keep}: BUFFER-X Rank-1 {np.mean(r1s):.3f}±{np.std(r1s):.3f}  "
                  f"AUC {np.mean(aucs):.3f}  ({time.time()-t_start:.0f}s)", flush=True)
    res["seconds"] = round(time.time() - t_start, 1)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
