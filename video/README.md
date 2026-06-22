# Showcase video pipeline

A free, mostly-local pipeline for a 1–3 minute social-media showcase of ToothPrint (researched
2026-06; all tools verified against primary sources). The box needs only `ffmpeg`, `python`, and
optionally `google-chrome` — already present here.

## What's built and ready

**`make_showcase.py`** — a narrated reel from the committed result figures. Runs with no GUI and no
API key:

```bash
pip install edge-tts            # free TTS, no key (uses Microsoft Edge's backend)
python video/make_showcase.py   # -> docs/toothprint_showcase.mp4 (1080p, ~84s, +faststart)
```

It renders title/end cards in the project palette, narrates each section with `edge-tts`
(`en-US-GuyNeural`), holds each figure for exactly its narration length, and muxes to a
social-ready H.264 MP4. Edit the `segments` list to change the script or figures.

*Fully offline / commercial-safe swap:* replace `edge-tts` with **Kokoro-82M** (Apache-2.0, local
CPU TTS: `sudo apt install espeak-ng && pip install "kokoro>=0.9.4" soundfile`). Avoid Coqui/XTSS —
non-commercial license.

## Extending it (CLI demo + app capture)

The reel covers the results. To also show the **terminal** and the **desktop app**:

**1. Terminal demo — VHS** (Charmbracelet; code-driven `.tape` → MP4, reproducible):

```bash
# install (Debian/Ubuntu): needs ttyd + ffmpeg on PATH
curl -fsSL https://repo.charm.sh/apt/gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/charm.gpg
echo "deb [signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" | sudo tee /etc/apt/sources.list.d/charm.list
sudo apt update && sudo apt install vhs ttyd ffmpeg
vhs video/demo.tape            # -> video/cli_demo.mp4
```

**2. Desktop app + browser** — screen capture (VHS can't grab GUIs):

```bash
echo $XDG_SESSION_TYPE                              # x11 or wayland?
# X11:    sudo apt install obs-studio   (multi-source GUI)  — or scriptable region grab:
ffmpeg -f x11grab -r 30 -s 1280x720 -i :0.0+100,200 -c:v libx264 -crf 18 -pix_fmt yuv420p app.mp4
# Wayland: wf-recorder -g "$(slurp)" -f app.mp4   (or the Kooha flatpak)
```

**3. Stitch everything** — concat the sections with ffmpeg (see `make_showcase.py` for the
image+audio clip pattern; `-movflags +faststart` is the key flag for X/LinkedIn streaming).

## Why these tools

No official Claude Code "make a video" skill exists (the plugin marketplace has none; the Mux
`agent-video` MCP needs paid ElevenLabs/Mux keys and records only web pages). VHS + OBS/wf-recorder
+ ffmpeg + edge-tts is the current de-facto free dev-showcase stack.
