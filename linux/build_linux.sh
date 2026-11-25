#!/usr/bin/env bash
# Build the FlashAttention Turing V2.3.2 wheel for Linux x86_64 (sm75).
# Applies the V2.0 -> V2.2 -> V2.3 -> V2.3.2 source overlays, in order, on top of
# ssiu/flash-attention-turing at BASE_COMMIT, then builds with the V2.3.2 build flags
# (the same ones as V2.3.2_d96n128/RUN_V23_BUILD.bat).
#
# Usage: linux/build_linux.sh <out-dir>
# Needs: nvcc (CUDA 13.0 for torch cu130), a Python with torch already installed, git.
set -euo pipefail
OUT="$(realpath -m "${1:-dist}")"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BASE_REPO="${BASE_REPO:-https://github.com/ssiu/flash-attention-turing}"
BASE_COMMIT="${BASE_COMMIT:-9ef98fcb}"
OVERLAYS="${OVERLAYS:-V2.0/source_overlay V2.2/source_overlay_v22 V2.3/source_overlay_v23 V2.3.2_d96n128/source_overlay_v23}"
PY="${PYTHON:-python}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone --quiet "$BASE_REPO" "$WORK/src"
git -C "$WORK/src" checkout --quiet "$BASE_COMMIT"
git -C "$WORK/src" submodule update --init --depth 1 csrc/cutlass
for o in $OVERLAYS; do
  echo "overlay: $o"
  cp -R "$HERE/$o/." "$WORK/src/"
done
# Linux packaging: ship the Python interface next to the extension and give the wheel a real version.
VERSION="${FA_TURING_VERSION:-2.3.2}"
TORCH_TAG="$("$PY" -c 'import torch;v=torch.__version__.split("+");print("torch"+v[0].replace(".","")[:3]+(("."+v[1]) if len(v)>1 else ""))')"
"$PY" - "$WORK/src/setup.py" "$VERSION+$TORCH_TAG" <<'PYEOF'
import sys, re
p, ver = sys.argv[1], sys.argv[2]
s = open(p).read()
# torch >= 2.13 headers require C++20 (the overlays pass -std=c++17, which suits the torch 2.11 Windows builds).
s = s.replace('"-std=c++17"', '"-std=c++20"')
s = s.replace('name="flash_attn_turing",', f'name="flash_attn_turing",\n    version="{ver}",\n    py_modules=["flash_attention_interface"],', 1)
open(p, "w").write(s)
PYEOF

export TORCH_CUDA_ARCH_LIST="7.5"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export FLASH_TURING_NATIVE_D32=1 FLASH_TURING_SEPV=1 FLASH_TURING_FASTSM=1
export FLASH_TURING_SMALLM=2 FLASH_TURING_CAUSAL_REVERSE=1 FLASH_TURING_D96_M64_N128=1
mkdir -p "$OUT"
cd "$WORK/src"
"$PY" -m pip wheel --no-deps --no-build-isolation -w "$OUT" -v . 2>&1 | grep -Ev "^\s*(ptxas info|ptxas  )" || true
ls -la "$OUT"/flash_attn_turing-*.whl
