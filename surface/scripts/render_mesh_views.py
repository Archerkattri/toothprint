#!/usr/bin/env python3
"""Render the 5 protocol views of a real Poseidon3D mesh into a montage PNG.

Usage:
    python scripts/render_mesh_views.py \
        --mesh data/poseidon3d/extracted/data/000062/000062_MODEL_mandible.stl \
        --out docs/dmc_5view_render.png --resolution 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dentalmapcert.render import render_5_views, rendered_view_to_pil


def main() -> None:
    ap = argparse.ArgumentParser(description="Render 5-view montage of a dental mesh")
    ap.add_argument("--mesh", default="data/poseidon3d/extracted/data/000062/000062_MODEL_mandible.stl")
    ap.add_argument("--out", type=Path, default=Path("docs/dmc_5view_render.png"))
    ap.add_argument("--resolution", type=int, default=256)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    views = render_5_views(args.mesh, resolution=args.resolution)
    imgs = [(name, rendered_view_to_pil(v)) for name, v in views.items()]
    w, h = imgs[0][1].size
    pad, label_h = 6, 18
    cols = len(imgs)
    montage = Image.new("RGB", (cols * w + (cols - 1) * pad, h + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(montage)
    for i, (name, img) in enumerate(imgs):
        x = i * (w + pad)
        montage.paste(img, (x, label_h))
        draw.text((x + 2, 2), name, fill=(230, 230, 230))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.out)
    print(f"Saved {args.out} ({montage.size[0]}x{montage.size[1]})")


if __name__ == "__main__":
    main()
