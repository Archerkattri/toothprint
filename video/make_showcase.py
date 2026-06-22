#!/usr/bin/env python3
"""Build a fast-paced, SILENT social-media reel that shows HOW ToothPrint works and HOW WELL.

It leads with the real dental visuals — a rotating intraoral scan, the genuine-vs-impostor
registration, the alignment sweep, the bone-level change certificate, and photo-to-mesh
reconstruction — sped up and captioned, interleaved with ANIMATED result charts (the
partial-overlap bars grow; the headline numbers count up). No narration: you add the caption when
posting. Quick fades between segments keep it punchy but readable. Output:
docs/toothprint_showcase.mp4 (1080p, H.264, no audio).

Requires: ffmpeg + matplotlib. Run from the repo root: `python video/make_showcase.py`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
WORK = ROOT / "video" / "_reel"; WORK.mkdir(parents=True, exist_ok=True)
OUT = DOCS / "toothprint_showcase.mp4"
INK, TEAL, PAPER, FAINT, AMBER, GREEN, ROSE = "#1b2329", "#11505f", "#f7f6f3", "#8a969c", "#c98a1e", "#16a34a", "#e11d48"
FPS = 30


def ease(t):
    return 1 - (1 - t) ** 3


def label_card(title, sub, path, sub_color=TEAL):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 0.006, 1, color=TEAL))
    ax.text(0.5, 0.935, title, ha="center", fontsize=42, fontweight="bold", color=INK, family="serif")
    if sub:
        ax.text(0.5, 0.05, sub, ha="center", fontsize=27, color=sub_color, family="sans-serif")
    fig.savefig(path, facecolor=PAPER); plt.close(fig)


def title_card(path, title, lines, sub):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=92, fontweight="bold", color=INK, family="serif")
    ax.plot([0.43, 0.57], [0.5, 0.5], color=TEAL, lw=2.5)
    for i, ln in enumerate(lines):
        ax.text(0.5, 0.4 - i * 0.07, ln, ha="center", fontsize=30, color=TEAL, family="sans-serif")
    ax.text(0.5, 0.12, sub, ha="center", fontsize=18, color=FAINT, family="monospace")
    fig.savefig(path, facecolor=PAPER); plt.close(fig)


def still_seg(png, out, dur):
    subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", str(png), "-t", f"{dur}", "-r", str(FPS),
                    "-vf", f"fade=t=in:st=0:d=0.3,fade=t=out:st={dur-0.3:.2f}:d=0.3,format=yuv420p",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True)


def gif_seg(gif, title, sub, out, dur, speed):
    bg = WORK / f"bg_{out.stem}.png"; label_card(title, sub, bg)
    fc = (f"[1:v]setpts=PTS/{speed},scale=1640:760:force_original_aspect_ratio=decrease[g];"
          f"[0:v][g]overlay=(W-w)/2:(H-h)/2+18,"
          f"fade=t=in:st=0:d=0.18,fade=t=out:st={dur-0.18:.2f}:d=0.18,format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", str(bg), "-stream_loop", "-1", "-i", str(gif),
                    "-filter_complex", fc, "-map", "[v]", "-t", f"{dur}", "-r", str(FPS),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True)


def bars_seg(out, dur):
    methods = ["Rigid\nGICP", "Crop-hardened\nembedding", "CorrNet\n(learned correspondence)"]
    targets, cols = [0.23, 0.635, 0.87], [FAINT, TEAL, GREEN]
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.10, right=0.96)
    fig.suptitle("Identity with HALF the teeth missing", fontsize=46, fontweight="bold", color=INK, family="serif", y=0.95)
    fig.text(0.5, 0.85, "learned point-correspondence holds where rigid registration collapses — ~3.8×",
             ha="center", fontsize=25, color=TEAL)
    ax.set_ylim(0, 1.0); ax.set_xlim(-0.65, 2.65); ax.set_xticks(range(3)); ax.set_xticklabels(methods, fontsize=24)
    ax.set_ylabel("Rank-1 identification", fontsize=24); ax.tick_params(labelsize=18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    bars = ax.bar(range(3), [0, 0, 0], color=cols, width=0.62)
    txt = [ax.text(i, 0, "", ha="center", fontsize=34, fontweight="bold", color=cols[i], family="monospace") for i in range(3)]
    n = int(dur * FPS)

    def upd(f):
        t = ease(min(1, f / (n * 0.6)))
        for i, b in enumerate(bars):
            h = targets[i] * t; b.set_height(h); txt[i].set_y(h + 0.02); txt[i].set_text(f"{h:.2f}")
        return bars
    FuncAnimation(fig, upd, frames=n).save(str(out), writer=FFMpegWriter(fps=FPS, bitrate=6000), dpi=100); plt.close(fig)


def stats_seg(out, dur):
    items = [("3D Rank-1", 0.995, "{:.3f}", TEAL), ("2D radiographs", 1.0, "{:.3f}", TEAL),
             ("50% tooth loss", 0.87, "{:.2f}", GREEN), ("False-alarm", 0.0, "{:.0%}", TEAL)]
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor(PAPER); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("How well — every verdict conformally certified", fontsize=44, fontweight="bold", color=INK, family="serif", y=0.84)
    num = [ax.text(0.135 + i * 0.243, 0.52, "", ha="center", fontsize=66, fontweight="bold", color=c, family="monospace") for i, (_, _, _, c) in enumerate(items)]
    for i, (lab, _, _, _) in enumerate(items):
        ax.text(0.135 + i * 0.243, 0.36, lab, ha="center", fontsize=23, color=INK, family="sans-serif")
    ax.text(0.5, 0.16, "conformal — a distribution-free false-alarm bound on every verdict", ha="center", fontsize=24, color=FAINT, family="sans-serif")
    n = int(dur * FPS)

    def upd(f):
        for i, (_, tg, fmt, _) in enumerate(items):
            t = ease(max(0, min(1, (f / n - i * 0.1) / 0.45)))
            num[i].set_text(fmt.format(tg * t))
        return num
    FuncAnimation(fig, upd, frames=n).save(str(out), writer=FFMpegWriter(fps=FPS, bitrate=6000), dpi=100); plt.close(fig)


def main():
    title_card(WORK / "title.png", "ToothPrint", ["recognise a person by their teeth,", "and certify what changed"],
               "open source · conformal-certified · github.com/Archerkattri/toothprint")
    title_card(WORK / "end.png", "Open source.", ["Free for everyone — including hospitals.", "github.com/Archerkattri/toothprint"],
               "validated in simulation · real longitudinal data is the next frontier")
    segs = []

    def add(seg):
        segs.append(seg)

    still_seg(WORK / "title.png", WORK / "s0.mp4", 2.6); add(WORK / "s0.mp4")
    gif_seg(DOCS / "input_arch_spin.gif", "The input — a real 3D dental scan", "every crown, cusp, and margin is individual", WORK / "s1.mp4", 3.2, 1.4); add(WORK / "s1.mp4")
    gif_seg(DOCS / "identity_match.gif", "Identity — best rigid fit to each gallery arch", "a genuine re-scan snaps on (0.05 mm) · a stranger floats off (4 mm)", WORK / "s2.mp4", 4.2, 1.15); add(WORK / "s2.mp4")
    gif_seg(DOCS / "alignment_proof.gif", "The fit is exact", "sub-0.1 mm point-to-surface — below sensor noise", WORK / "s3.mp4", 3.2, 1.35); add(WORK / "s3.mp4")
    bars_seg(WORK / "s4.mp4", 4.2); add(WORK / "s4.mp4")
    gif_seg(DOCS / "change_measurement.gif", "Change — certified bone-level recession", "sub-pixel registration between two visits, certified past threshold", WORK / "s5.mp4", 4.2, 1.0); add(WORK / "s5.mp4")
    gif_seg(DOCS / "recon_turntable.gif", "No scanner? Photos → a dentist-usable mesh", "2D Gaussian Splatting · ~0.3 mm median", WORK / "s6.mp4", 3.4, 1.4); add(WORK / "s6.mp4")
    stats_seg(WORK / "s7.mp4", 4.0); add(WORK / "s7.mp4")
    still_seg(WORK / "end.png", WORK / "s8.mp4", 2.8); add(WORK / "s8.mp4")

    inputs = []
    for s in segs:
        inputs += ["-i", str(s)]
    fc = "".join(f"[{i}:v]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[v]"
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[v]",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
                    "-movflags", "+faststart", str(OUT)], check=True, capture_output=True)
    d = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(OUT)],
                       capture_output=True, text=True).stdout.strip()
    print(f"wrote {OUT}  ({float(d):.0f}s, silent)")


if __name__ == "__main__":
    main()
