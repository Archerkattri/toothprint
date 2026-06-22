# Showcase video pipeline

A free, mostly-local pipeline for a 1–3 minute social-media showcase of ToothPrint (researched
2026-06; all tools verified against primary sources). The box needs only `ffmpeg`, `python`, and
optionally `google-chrome` — already present here.

## What's built and ready

**`make_showcase.py`** — a fast-paced, **silent** social reel that shows *how* ToothPrint works and
*how well*, from the real dental visuals plus animated charts. No GUI, no API key, no narration (you
add your own caption when posting to LinkedIn / X):

```bash
python video/make_showcase.py   # -> docs/toothprint_showcase.mp4 (1080p, ~32s, silent, +faststart)
```

It speeds up and captions the committed dental GIFs — the rotating intraoral scan, the
genuine-vs-impostor registration, the alignment sweep, the bone-level change certificate, and
photo-to-mesh reconstruction — and interleaves **animated** result charts (the partial-overlap bars
grow; the headline numbers count up), with quick fades between segments to stay punchy but readable.
Edit the `segs` list in `main()` to change the order, clips, or captions.

*Want narration instead of on-post captions?* `pip install edge-tts` (free, no key) and generate a
voiceover (`edge-tts --voice en-US-GuyNeural --text "..." --write-media n.mp3`), or **Kokoro-82M**
for fully-offline / commercial-safe TTS — then mux it over the reel with ffmpeg.

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
