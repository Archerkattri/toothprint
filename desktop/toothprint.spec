# PyInstaller spec — bundles ToothPrint Studio into a single-folder desktop app.
# Build ON the target OS (PyInstaller does not cross-compile):
#   pip install -e ".[api,io,desktop,recon]" pyinstaller
#   pyinstaller desktop/toothprint.spec
# Output: dist/ToothPrint/  (run the ToothPrint executable inside).
#
# The certification core (numpy/scipy/opencv/open3d) is heavy; open3d in particular
# needs its data files collected. Adjust hiddenimports if your platform's webview
# backend differs (Linux: pywebview[qt] or [gtk]; Windows: EdgeChromium; macOS: WebKit).
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = [("../web", "web")]                       # the Studio UI, served locally
for pkg in ("open3d", "trimesh", "pydicom", "nibabel"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

hiddenimports = (
    collect_submodules("uvicorn") + collect_submodules("toothprint")
    + ["api.main", "anyio", "pylibjpeg", "pylibjpeg_libjpeg", "pylibjpeg_openjpeg"]
)

a = Analysis(["app.py"], pathex=[".."], binaries=[], datas=datas,
             hiddenimports=hiddenimports, hookspath=[], runtime_hooks=[],
             excludes=["tkinter"], cipher=block_cipher, noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ToothPrint",
          console=False, disable_windowed_traceback=False,
          icon=None)                              # drop a .ico/.icns here per platform
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name="ToothPrint")
