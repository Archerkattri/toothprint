#!/usr/bin/env python3
"""Validate 2D dental identification on real DenPAR radiograph landmarks.

Gallery = each subject's landmark constellation. Query = the same subject
re-acquired (similarity reposition + magnification + landmark jitter, the
acquisition model). Identify each query against the gallery by smallest
similarity-ICP residual; report Rank-1 accuracy and genuine/impostor separation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "change"))  # dcc package
from toothid.landmark_id import constellation, residual_matrix
from toothid.mesh_id import identification_metrics


def _perturb(c, rng, mag, jitter):
    ang = rng.uniform(-0.15, 0.15)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    scale = 1.0 + rng.uniform(-mag, mag)
    t = rng.uniform(-20, 20, 2)
    return (scale * c @ R.T) + t + rng.normal(0, jitter, c.shape)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../change/data/denpar/extracted/Dataset")
    p.add_argument("--output", default="outputs/landmark_identification")
    p.add_argument("--n-subjects", type=int, default=40)
    p.add_argument("--min-teeth", type=int, default=4)
    p.add_argument("--magnification", type=float, default=0.08)
    p.add_argument("--jitter", type=float, default=3.0)
    args = p.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"ERROR: DenPAR not found at {root}", file=sys.stderr)
        return 1
    from dcc.data.denpar_adapter import RealDenparAdapter

    gallery, labels = [], []
    for rec in RealDenparAdapter(root).records("test"):
        teeth = [t for t in rec.annotation_dict.get("teeth", []) if t.get("cej") and t.get("crest_line")]
        if len(teeth) < args.min_teeth:
            continue
        try:
            c = constellation({"teeth": teeth})
        except ValueError:
            continue
        gallery.append(c); labels.append(rec.image_id)
        if len(gallery) >= args.n_subjects:
            break
    print(f"Enrolled {len(gallery)} subjects' landmark constellations.")

    rng = np.random.default_rng(0)
    queries = [_perturb(c, rng, args.magnification, args.jitter) for c in gallery]
    rmat = residual_matrix(queries, gallery)
    metrics = identification_metrics(rmat, labels, labels)

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(
        {**metrics, "residual": rmat.tolist(), "labels": labels}, indent=2))

    print(f"\n=== 2D dental identification (real DenPAR constellations) ===")
    print(f"  Rank-1 accuracy:           {metrics['rank1_accuracy']:.3f} "
          f"({metrics['n_query']} queries vs {metrics['n_gallery']} gallery)")
    print(f"  genuine residual mean/max: {metrics['genuine_rmse_mean']:.2f} / {metrics['genuine_rmse_max']:.2f} px")
    print(f"  impostor residual mean/min:{metrics['impostor_rmse_mean']:.2f} / {metrics['impostor_rmse_min']:.2f} px")
    print(f"  decidability d':           {metrics['decidability_dprime']:.2f}")
    print(f"\n  Metrics: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
