# ToothPrint Studio — desktop app

A native desktop app (Linux · Windows · macOS) for the three certificates, with
drag-and-drop ingest of every common dental format. The UI is a local single page
(`web/studio.html`) served by the in-process FastAPI backend and shown in a native
webview window — no browser, no cloud, files never leave the machine.

## Run from source

```bash
pip install -e ".[api,io,desktop]"     # add ",recon" for the photo→mesh front end
python -m desktop.app
```

A window opens on **ToothPrint Studio**. On a headless machine (no display/webview) it
prints a local URL and falls back to your browser.

What it does:
- **Drop a file** → it's parsed safely (DICOM / STL / PLY / OBJ / 3MF / NIfTI /
  PNG·JPG·TIFF, detected by content) and the specimen panel shows kind, format,
  dimensions, pixel spacing or mesh extent.
- **Identity** → drop a query arch + gallery arches; the query is best-rigid-fit to each
  and the closest surface wins, with a same-person / no-match verdict.
- **Change / Surface** → enter the measurement + calibration; the conformal certificate
  returns changed / stable / uncertain with the interval, and the **seal stamps only
  when the interval clears the threshold** (the verdict carries an α-bounded false-alarm
  rate, not a guess).

## Build native installers

PyInstaller does **not** cross-compile — build each OS on that OS (or a CI matrix /
VM). On the target platform:

```bash
pip install -e ".[api,io,desktop,recon]" pyinstaller
pyinstaller desktop/toothprint.spec
# -> dist/ToothPrint/ToothPrint(.exe / .app)
```

Per-OS notes:
- **Linux**: needs a webview backend — `pip install pywebview[qt]` (PySide6) or
  `pywebview[gtk]`. Package `dist/ToothPrint/` as an AppImage/`.deb` if you want an installer.
- **Windows**: uses the built-in EdgeChromium WebView2 runtime (present on Win10/11).
  Wrap `dist\ToothPrint\` with Inno Setup or NSIS for an `.exe` installer; add a `.ico`.
- **macOS**: produces `ToothPrint.app`; add a `.icns`, then `codesign` + `notarize` for
  distribution. Build on both arm64 and x86_64 for a universal app.

open3d/torch make the bundle large (hundreds of MB); the `recon` extra is optional and
only needed for the photo→mesh feature. The certificate + ingest features work without it.

> Research prototype, not a cleared medical device — see the Clinical readiness
> section of the [main README](../README.md#clinical-readiness).
