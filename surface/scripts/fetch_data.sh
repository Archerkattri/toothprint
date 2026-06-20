#!/usr/bin/env bash
#
# fetch_data.sh — download + extract DentalMapCert reference datasets into the
# exact paths the scripts/loaders expect under the gitignored data/ directory.
#
# This script ONLY downloads and extracts data. It does NOT run any
# reconstruction / render / GPU job, and it does NOT download model weights
# (VGGT/DUSt3R fetch their own weights at run time on the GPU machine).
#
# See docs/DATA.md for the expected layout, sources, and the reproduce commands.
#
# Datasets:
#   Poseidon3D   STL meshes      -> data/poseidon3d/extracted/data/   (Zenodo 15608906, CC-BY-4.0)
#   3DTeethLand  landmark JSON   -> data/teeth3ds/extracted/          (OSF um96h, training split)
#   Teeth3DS     OBJ meshes      -> data/teeth3ds/obj/ + labels/      (Grand Challenge; manual)
#
# Archive URLs are read from env vars so you can override them without editing
# this file (record assets are occasionally re-versioned). Confirm the current
# filenames on the source pages before running.
#
# Usage:
#   bash scripts/fetch_data.sh              # all openly-downloadable datasets
#   bash scripts/fetch_data.sh --poseidon   # only Poseidon3D
#   bash scripts/fetch_data.sh --teethland  # only 3DTeethLand
#
set -euo pipefail

# Resolve repo root (this script lives in <repo>/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"

# Override these to point at the exact current archive assets.
POSEIDON3D_URL="${POSEIDON3D_URL:-https://zenodo.org/records/15608906/files/poseidon3d.zip}"
TEETHLAND_URL="${TEETHLAND_URL:-https://osf.io/um96h/download}"

DO_POSEIDON=0
DO_TEETHLAND=0
case "${1:---all}" in
  --all)       DO_POSEIDON=1; DO_TEETHLAND=1 ;;
  --poseidon)  DO_POSEIDON=1 ;;
  --teethland) DO_TEETHLAND=1 ;;
  -h|--help)
    sed -n '2,/^set -euo/{/^set -euo/!p}' "$0" | sed 's/^# \{0,1\}//'
    exit 0 ;;
  *)
    echo "Unknown option: $1 (use --all | --poseidon | --teethland | --help)" >&2
    exit 2 ;;
esac

_have() { command -v "$1" >/dev/null 2>&1; }

_download() {
  # _download <url> <dest-file>
  local url="$1" dest="$2"
  if [[ -f "${dest}" ]]; then
    echo "  already downloaded: ${dest}"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  echo "  downloading ${url}"
  if _have curl; then
    curl -L --fail -o "${dest}" "${url}"
  elif _have wget; then
    wget -O "${dest}" "${url}"
  else
    echo "ERROR: need curl or wget to download." >&2
    exit 1
  fi
}

_extract() {
  # _extract <archive> <dest-dir>
  local archive="$1" dest="$2"
  mkdir -p "${dest}"
  echo "  extracting ${archive} -> ${dest}"
  case "${archive}" in
    *.zip)            unzip -q -o "${archive}" -d "${dest}" ;;
    *.tar.gz|*.tgz)   tar -xzf "${archive}" -C "${dest}" ;;
    *.tar)            tar -xf  "${archive}" -C "${dest}" ;;
    *)
      echo "ERROR: unknown archive type: ${archive}" >&2
      exit 1 ;;
  esac
}

if [[ "${DO_POSEIDON}" == "1" ]]; then
  echo "[Poseidon3D] -> data/poseidon3d/extracted/data/"
  P_RAW="${DATA_DIR}/poseidon3d/raw"
  P_EXTRACTED="${DATA_DIR}/poseidon3d/extracted"
  _download "${POSEIDON3D_URL}" "${P_RAW}/poseidon3d_archive"
  _extract  "${P_RAW}/poseidon3d_archive" "${P_EXTRACTED}"
  # Expected after extraction: ${P_EXTRACTED}/data/metadata.json and per-case dirs.
  if [[ ! -f "${P_EXTRACTED}/data/metadata.json" ]]; then
    echo "  NOTE: ${P_EXTRACTED}/data/metadata.json not found after extraction." >&2
    echo "  The archive layout may differ; move the extracted 'data/' dir so that" >&2
    echo "  ${P_EXTRACTED}/data/metadata.json exists (see docs/DATA.md)." >&2
  fi
fi

if [[ "${DO_TEETHLAND}" == "1" ]]; then
  echo "[3DTeethLand] -> data/teeth3ds/extracted/ (upper/ + lower/)"
  T_RAW="${DATA_DIR}/teeth3ds/raw"
  T_EXTRACTED="${DATA_DIR}/teeth3ds/extracted"
  _download "${TEETHLAND_URL}" "${T_RAW}/teethland_archive"
  _extract  "${T_RAW}/teethland_archive" "${T_EXTRACTED}"
  if [[ ! -d "${T_EXTRACTED}/upper" && ! -d "${T_EXTRACTED}/lower" ]]; then
    echo "  NOTE: neither ${T_EXTRACTED}/upper nor ${T_EXTRACTED}/lower found." >&2
    echo "  Re-arrange the extracted landmark JSONs into upper/ and lower/" >&2
    echo "  case directories (see docs/DATA.md)." >&2
  fi
fi

# Teeth3DS OBJ meshes are gated behind Grand Challenge registration and cannot
# be fetched non-interactively. Document the manual step rather than failing.
cat <<'EOF'

[Teeth3DS OBJ] manual step (Grand Challenge registration required)
  1. Register at https://teeth3ds.grand-challenge.org/ and download the OBJ meshes.
  2. Place them so that:
       data/teeth3ds/obj/*.obj
       data/teeth3ds/labels/<paired label files>
     (Teeth3DSLoader reads <root>/obj and <root>/labels.)

Done. Validate paths with the snippet in docs/DATA.md before running anything.
No reconstruction/GPU job was started by this script.
EOF
