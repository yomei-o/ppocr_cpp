#!/bin/sh
# MSVC build without vcvars (vcvarsall hangs in this environment — see yolov8_cpp RESUME notes).
# Finds the newest VS2022 MSVC toolset + Windows SDK and sets INCLUDE/LIB/PATH by hand.
#
#   sh build/cc.sh pure/ppocr.cpp -o ppocr.exe            # release
#   EXTRA="-DUSE_EIGEN -arch:AVX2" sh build/cc.sh ...    # Eigen CPU fast path
set -e

VSROOT="/c/Program Files/Microsoft Visual Studio/2022"
MSVC_DIR=$(ls -d "$VSROOT"/*/VC/Tools/MSVC/* 2>/dev/null | sort -V | tail -1)
SDK_INC=$(ls -d "/c/Program Files (x86)/Windows Kits/10/Include"/* 2>/dev/null | sort -V | tail -1)
SDK_LIB=$(ls -d "/c/Program Files (x86)/Windows Kits/10/Lib"/* 2>/dev/null | sort -V | tail -1)
[ -n "$MSVC_DIR" ] || { echo "no MSVC toolset found under $VSROOT"; exit 1; }
[ -n "$SDK_INC" ] || { echo "no Windows SDK found"; exit 1; }

BIN="$MSVC_DIR/bin/Hostx64/x64"
export INCLUDE="$(cygpath -w "$MSVC_DIR/include");$(cygpath -w "$SDK_INC/ucrt");$(cygpath -w "$SDK_INC/um");$(cygpath -w "$SDK_INC/shared");$(cygpath -w "$SDK_INC/winrt")"
export LIB="$(cygpath -w "$MSVC_DIR/lib/x64");$(cygpath -w "$SDK_LIB/ucrt/x64");$(cygpath -w "$SDK_LIB/um/x64")"
export PATH="$BIN:$PATH"

SRC="$1"; shift
OUT="ppocr.exe"
if [ "$1" = "-o" ]; then OUT="$2"; shift 2; fi

mkdir -p scratch
# shell32.lib: main() calls CommandLineToArgvW (UTF-8 argv). Some SDK/toolset combinations pull it in
# implicitly and some do not (14.31 + SDK 10.0.19041 does not), so name it rather than depend on that.
cl.exe //nologo //std:c++20 //O2 //EHsc //utf-8 //Zc:preprocessor //DNOMINMAX \
  //I pure //I pure/third_party //I pure/third_party/eigen_flat \
  $EXTRA "$@" "$(cygpath -w "$SRC")" shell32.lib //Fo:scratch\\ //Fe:"$(cygpath -w "$OUT")"
echo "built $OUT (MSVC $(basename "$MSVC_DIR"), SDK $(basename "$SDK_INC"))"
