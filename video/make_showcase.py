#!/usr/bin/env python3
"""Build a narrated social-media showcase reel of ToothPrint from the committed result figures.

edge-tts (free, no API key) narrates each section; ffmpeg holds the matching figure for exactly
that long, letterboxed to 1920x1080, with the narration muxed in. Title / end cards are rendered
with matplotlib in the project's restrained palette. Output: docs/toothprint_showcase.mp4 (1080p
H.264, +faststart — ready for YouTube / X / LinkedIn). Each segment is a self-contained clip
(image + narration + a short tail) so audio and video stay in sync through the concat.

Requires: `pip install edge-tts`, ffmpeg, matplotlib. Run from the repo root.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
WORK = ROOT / "video" / "_work"; WORK.mkdir(parents=True, exist_ok=True)
OUT = DOCS / "toothprint_showcase.mp4"
VOICE = "en-US-GuyNeural"
INK, TEAL, PAPER, FAINT = "#1b2329", "#11505f", "#f7f6f3", "#8a969c"


def card(path, title, lines, sub=""):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.63, title, ha="center", va="center", fontsize=60, fontweight="bold", color=INK, family="serif")
    ax.plot([0.43, 0.57], [0.55, 0.55], color=TEAL, lw=2.5)
    for i, ln in enumerate(lines):
        ax.text(0.5, 0.45 - i * 0.075, ln, ha="center", fontsize=27, color=TEAL, family="sans-serif")
    if sub:
        ax.text(0.5, 0.1, sub, ha="center", fontsize=17, color=FAINT, family="monospace")
    fig.savefig(path, facecolor=PAPER); plt.close(fig)


def dur(mp3):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "json", str(mp3)],
                         capture_output=True, text=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def main():
    card(WORK / "title.png", "ToothPrint",
         ["recognise a person by their teeth,", "and certify what changed"],
         "open source · conformal-certified · github.com/Archerkattri/toothprint")
    card(WORK / "end.png", "Validated in simulation.",
         ["The next frontier is real longitudinal data.", "Free for everyone — including hospitals."],
         "github.com/Archerkattri/toothprint · PolyForm Noncommercial · krishiattriwork@gmail.com")

    segments = [
        (WORK / "title.png", "ToothPrint recognises a person by their teeth, and certifies whether their teeth have changed."),
        (DOCS / "results_panel.png", "The dental arch is an individual tooth print. Given a 3D scan, ToothPrint finds the matching person with rank-one accuracy of point nine nine five. And, uniquely, even when half the teeth are missing, a learned point-correspondence matcher holds point eight seven, where every rigid method collapses to point two three."),
        (DOCS / "det_curves.png", "Every verdict is conformal. It carries a distribution-free false-alarm bound, firing only when it is sure, and abstaining when it is not. These are the first full detection-error tradeoff curves for dental identity."),
        (DOCS / "change_certificate_v2.png", "The same engine certifies bone-level change between two radiographs, measured differentially and certified only once it clears a clinical threshold."),
        (DOCS / "surface_certificate_v2.png", "And it certifies three-D surface change against the reconstruction's own noise, flagging a real lesion, but never the scanner."),
        (DOCS / "studio.png", "It reads every format a clinic has: DICOM, intraoral scans, and C-B-C-T. It runs as a cross-platform desktop app that plays the patient's video and exports a full PDF report."),
        (WORK / "end.png", "It is open source under a non-commercial license, free for hospitals. Every number is validated in simulation. Real longitudinal data is the next frontier."),
    ]

    clips = []
    for i, (img, text) in enumerate(segments):
        mp3 = WORK / f"a{i}.mp3"
        subprocess.run(["edge-tts", "--voice", VOICE, "--text", text, "--write-media", str(mp3)], check=True, capture_output=True)
        d = dur(mp3) + 0.8
        clip = WORK / f"seg{i}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1", "-i", str(img), "-i", str(mp3), "-t", f"{d:.2f}",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white,setsar=1",
            "-af", "apad", "-r", "30", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", str(clip)], check=True, capture_output=True)
        clips.append(clip)
        print(f"  segment {i}: {d:.1f}s", flush=True)

    lst = WORK / "clips.txt"; lst.write_text("".join(f"file '{c}'\n" for c in clips))
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(OUT)], check=True, capture_output=True)
    print(f"wrote {OUT}  ({dur(OUT):.0f}s)")


if __name__ == "__main__":
    main()
