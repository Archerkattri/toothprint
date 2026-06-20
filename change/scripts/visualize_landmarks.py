#!/usr/bin/env python3
"""Overlay real perio-KPT landmarks (CEJ / crest / apex) on a real radiograph.

CEJ (cyan) and crest/bone-level (red) define the periodontal bone level whose
change the certificate measures; apex (green) anchors the tooth. Output is a
committed PNG for the README.

Usage:
    python scripts/visualize_landmarks.py \
        --data data/perio-kpt/extracted/perio_KPT \
        --out docs/dcc_landmarks_overlay.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dcc.data.perio_kpt_adapter import PerioKptAdapter


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay perio-KPT landmarks on a radiograph")
    ap.add_argument("--data", default="data/perio-kpt/extracted/perio_KPT")
    ap.add_argument("--out", type=Path, default=Path("docs/dcc_landmarks_overlay.png"))
    ap.add_argument("--max-width", type=int, default=900)
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    adapter = PerioKptAdapter(args.data)
    rec = None
    for r in adapter.records("baseline"):
        if Path(r.image_path).exists() and len(r.annotation_dict.get("teeth", [])) >= 3:
            rec = r
            break
    if rec is None:
        print("No suitable radiograph found", file=sys.stderr)
        sys.exit(1)

    img = Image.open(rec.image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    def dot(p, color, rad=7):
        x, y = p
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=color, outline=(255, 255, 255))

    n = 0
    for tooth in rec.annotation_dict["teeth"]:
        for p in tooth.get("cej", []):
            dot(p, (0, 200, 255)); n += 1            # CEJ
        for p in tooth.get("crest_line", []):
            dot(p, (255, 80, 80)); n += 1            # crest / bone level
        for p in tooth.get("apex", []):
            dot(p, (120, 255, 120)); n += 1          # apex

    draw.rectangle([8, 8, 260, 86], fill=(0, 0, 0))
    draw.ellipse([16, 18, 28, 30], fill=(0, 200, 255)); draw.text((36, 18), "CEJ", fill=(255, 255, 255))
    draw.ellipse([16, 42, 28, 54], fill=(255, 80, 80)); draw.text((36, 42), "crest (bone level)", fill=(255, 255, 255))
    draw.ellipse([16, 66, 28, 78], fill=(120, 255, 120)); draw.text((36, 66), "apex", fill=(255, 255, 255))

    img.thumbnail((args.max_width, args.max_width))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"Saved {args.out} with {n} landmarks ({img.size[0]}x{img.size[1]}) from {rec.image_path}")


if __name__ == "__main__":
    main()
