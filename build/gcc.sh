#!/bin/sh
# mingw / g++ build (w64devkit here).
#   sh build/gcc.sh pure/ppocr.cpp -o ppocr.exe
#   EXTRA="-fopenmp" sh build/gcc.sh ...                 # OpenMP
#   EXTRA="-DUSE_EIGEN -mavx2 -mfma" sh build/gcc.sh ...  # Eigen CPU fast path
set -e

# Below-normal priority so a parallel build does not make the rest of the desktop stutter.
. "$(dirname "$0")/lowpri.sh"
SRC="$1"; shift
OUT="ppocr.exe"
if [ "$1" = "-o" ]; then OUT="$2"; shift 2; fi
g++ -std=c++20 -O2 -Ipure -Ipure/third_party -Ipure/third_party/eigen_flat $EXTRA "$@" "$SRC" -o "$OUT"
echo "built $OUT ($(g++ --version | head -1))"
