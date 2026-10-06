#!/usr/bin/env bash
# Yahboom's Rosmaster_Lib is not on PyPI -- it ships as a zipped egg in the
# robot's system Python. Extract it into vendor/ so the container is
# self-contained and the pinned copy always matches this robot's hardware.
#
# Run once, on the robot:   bash tools/vendor_sdk.sh
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/vendor"
EGG=$(python3 -c 'import Rosmaster_Lib, os; print(os.path.dirname(os.path.dirname(Rosmaster_Lib.__file__)))')

if [ ! -f "$EGG" ]; then
  echo "error: expected a zipped egg, found '$EGG'" >&2
  exit 1
fi

mkdir -p "$DEST"
python3 - "$EGG" "$DEST" <<'PY'
import sys, zipfile
egg, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(egg) as z:
    names = [n for n in z.namelist() if n.startswith("Rosmaster_Lib/") and n.endswith(".py")]
    z.extractall(dest, members=names)
    print(f"extracted {len(names)} file(s) from {egg}")
PY

VERSION=$(basename "$EGG")
printf 'Vendored unmodified from the robot at %s\nSource egg: %s\nExtracted:  %s\n' \
  "$(date -Iseconds)" "$VERSION" "$(date -Iseconds)" > "$DEST/NOTICE"

echo "vendored into $DEST"
