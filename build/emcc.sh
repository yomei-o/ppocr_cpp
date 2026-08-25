#!/bin/sh
# Emscripten build for the browser demo. emsdk lives at C:\prog\emsdk\emsdk on this machine
# (override with EMSDK=...). Run `./emsdk install latest && ./emsdk activate latest` there once.
#
#   sh build/emcc.sh wasm/ppocr_wasm.cpp -o wasm/ppocr.js
set -e

# Below-normal priority so a parallel build does not make the rest of the desktop stutter.
. "$(dirname "$0")/lowpri.sh"
EMSDK="${EMSDK:-/c/prog/emsdk/emsdk}"
EMCC="$EMSDK/upstream/emscripten/emcc.py"
[ -f "$EMCC" ] || { echo "emcc.py not found at $EMCC — run: cd $EMSDK && ./emsdk install latest"; exit 1; }
export EM_CONFIG="$EMSDK/.emscripten"

SRC="$1"; shift
OUT="wasm/ppocr.js"
if [ "$1" = "-o" ]; then OUT="$2"; shift 2; fi

python "$EMCC" -std=c++20 -O3 -msimd128 -Ipure -Ipure/third_party -Ipure/third_party/eigen_flat \
  -s MODULARIZE=1 -s EXPORT_NAME=createPpocr -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","stringToUTF8","lengthBytesUTF8","HEAPU8","HEAPF32"]' \
  -s EXPORTED_FUNCTIONS='["_malloc","_free"]' \
  $EXTRA "$@" "$SRC" -o "$OUT"
echo "built $OUT"
