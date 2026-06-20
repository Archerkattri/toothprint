#!/usr/bin/env bash
# fetch_data.sh — verify + extract DentalChangeCert datasets into the paths the
# gate scripts expect. This script DOES NOT download anything: DenPAR must
# already be present at data/denpar/raw/ (the repo ships the raw zip there), and
# perio-KPT is access-gated and cannot be auto-fetched. See docs/DATA.md for the
# manual download steps and provenance.
#
# Usage:
#   scripts/fetch_data.sh                 # extract everything that is present
#   scripts/fetch_data.sh denpar          # extract only DenPAR
#   scripts/fetch_data.sh verify          # checksum-verify raw archives only
#
# Idempotent: re-running re-uses existing extracted trees unless --force is set.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DENPAR_ZIP="data/denpar/raw/DenPAR_Radiographs_Dataset.zip"
DENPAR_ZIP_SHA256="b9edb55020f2cb971ba771b4cf5e4b65c4abb4df957310bd2eccc83d5a08b072"
DENPAR_EXTRACT_DIR="data/denpar/extracted"          # -> data/denpar/extracted/Dataset/
DENPAR_EXPECT="data/denpar/extracted/Dataset"

PERIO_RAW_DIR="data/perio-kpt/raw"
PERIO_EXPECT="data/perio-kpt/extracted/perio_KPT"

FORCE=0
TARGET="all"
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    denpar|perio-kpt|verify|all) TARGET="$arg" ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

verify_denpar_zip() {
  if [[ ! -f "$DENPAR_ZIP" ]]; then
    echo "MISSING: $DENPAR_ZIP" >&2
    echo "  DenPAR is open (CC-BY-4.0). Download the archive and place it at" >&2
    echo "  $DENPAR_ZIP — see docs/DATA.md for the source URL/DOI." >&2
    return 1
  fi
  echo "Verifying $DENPAR_ZIP ..."
  local got; got="$(_sha256 "$DENPAR_ZIP")"
  if [[ "$got" != "$DENPAR_ZIP_SHA256" ]]; then
    echo "CHECKSUM MISMATCH for $DENPAR_ZIP" >&2
    echo "  expected $DENPAR_ZIP_SHA256" >&2
    echo "  got      $got" >&2
    return 1
  fi
  echo "  OK (sha256 matches)"
}

extract_denpar() {
  verify_denpar_zip
  if [[ -d "$DENPAR_EXPECT" && "$FORCE" -ne 1 ]]; then
    echo "DenPAR already extracted at $DENPAR_EXPECT (use --force to re-extract)."
    return 0
  fi
  echo "Extracting DenPAR into $DENPAR_EXTRACT_DIR/ ..."
  mkdir -p "$DENPAR_EXTRACT_DIR"
  unzip -q -o "$DENPAR_ZIP" -d "$DENPAR_EXTRACT_DIR"
  if [[ -d "$DENPAR_EXPECT/Training/Images" ]]; then
    echo "  OK -> $DENPAR_EXPECT"
  else
    echo "  WARNING: expected $DENPAR_EXPECT/Training/Images not found after extract" >&2
    return 1
  fi
}

handle_perio_kpt() {
  if [[ -d "$PERIO_EXPECT" ]]; then
    echo "perio-KPT present at $PERIO_EXPECT."
    return 0
  fi
  echo "perio-KPT NOT present (expected at $PERIO_EXPECT)."
  echo "  perio-KPT is an access-gated dataset and cannot be auto-fetched."
  echo "  1) Request/obtain the perio-KPT archive (see docs/DATA.md for provenance)."
  echo "  2) Place the archive under $PERIO_RAW_DIR/ ."
  echo "  3) Extract so the layout is:"
  echo "       $PERIO_EXPECT/0_Baseline/{images,labels}/"
  echo "       $PERIO_EXPECT/1_Experiment/standard_box/f*/{train,val}/{images,labels}/"
  echo "       $PERIO_EXPECT/1_Experiment/standard_box/holdout_test_standard_box/{images,labels}/"
  echo "       $PERIO_EXPECT/3_External_Set/standard_box/{images,labels}/"
  echo "  Gate-2 perio-KPT (scripts/run_gate2.py) and the M4 cross-source"
  echo "  experiment require this tree; until then run only the DenPAR gate."
  return 0
}

case "$TARGET" in
  verify)    verify_denpar_zip ;;
  denpar)    extract_denpar ;;
  perio-kpt) handle_perio_kpt ;;
  all)       extract_denpar; echo; handle_perio_kpt ;;
esac

echo
echo "Done. Next: see docs/DATA.md for the exact gate commands that regenerate outputs/."
