"""Cross-check the scratch C++ engine against onnxruntime + cv2 + pyclipper.

Two levels, and the first one is the one that matters:

  TENSOR   `ppocr dump` writes the input tensor the C++ side built and the output tensor it
           produced. This script feeds that same input to onnxruntime and diffs the outputs. That
           covers the interpreter end to end — 664 detector nodes, 746 recognizer nodes — against a
           completely independent implementation of the same graph. A mismatch here is a bug in
           pure/onnx_run.hpp, full stop.

  PIPELINE the two JSON outputs (`ppocr run --json` vs `ppocr_ref.py --json`) are matched box to
           box by IoU and compared as text. This layer is *allowed* to differ a little: the C++
           unclip grows the min-area rectangle where PaddleOCR offsets the polygon with pyclipper,
           so a box can land a pixel or two out and a marginal crop can read differently. The
           script reports how often, which is the number worth watching.

  python tools/parity.py --img assets/japan_2.jpg
"""
import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "ppocr.exe")


def read_dump(path):
    with open(path, "rb") as f:
        blob = f.read()
    assert blob[:4] == b"PPCR", "not a ppocr dump"
    off = 4
    (count,) = struct.unpack_from("<i", blob, off)
    off += 4
    out = []
    for _ in range(count):
        (nd,) = struct.unpack_from("<i", blob, off)
        off += 4
        dims = struct.unpack_from("<%di" % nd, blob, off)
        off += 4 * nd
        n = int(np.prod(dims))
        arr = np.frombuffer(blob, dtype=np.float32, count=n, offset=off).reshape(dims)
        off += 4 * n
        out.append(arr)
    return out


def quad_iou(a, b):
    """Axis-aligned IoU of the two quads' bounding boxes — enough to pair boxes up."""
    a, b = np.array(a), np.array(b)
    ax0, ay0, ax1, ay1 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
    bx0, by0, bx1, by1 = b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def tensor_level(img, model, onnx_path, limit):
    dump = os.path.join(ROOT, "scratch", "parity_%s.bin" % model)
    os.makedirs(os.path.dirname(dump), exist_ok=True)
    cmd = [EXE, "dump", "--img", img, "--model", model, "--out", dump, "--limit", str(limit)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print("  ppocr dump failed:", r.stdout.strip(), r.stderr.strip())
        return None
    x, y_cpp = read_dump(dump)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y_ref = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x)})[0]
    if y_ref.shape != y_cpp.shape:
        print("  %s: SHAPE MISMATCH cpp %s vs ort %s" % (model, y_cpp.shape, y_ref.shape))
        return None
    d = np.abs(y_ref.astype(np.float64) - y_cpp.astype(np.float64))
    scale = max(1e-9, float(np.abs(y_ref).max()))
    print("  %-3s in %-18s out %-20s max|d| %.3e  mean|d| %.3e  (rel %.2e)"
          % (model, "x".join(str(v) for v in x.shape), "x".join(str(v) for v in y_cpp.shape),
             d.max(), d.mean(), d.max() / scale))
    return float(d.max() / scale)


def pipeline_level(img, limit):
    cpp = subprocess.run([EXE, "run", "--img", img, "--json", "--limit", str(limit)],
                         cwd=ROOT, capture_output=True)
    txt = cpp.stdout.decode("utf-8", "replace").strip().splitlines()
    if not txt:
        print("  ppocr run produced nothing:", cpp.stderr.decode("utf-8", "replace")[:300])
        return
    a = json.loads(txt[-1])

    ref = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "ppocr_ref.py"),
                          "--img", img, "--json", "--limit", str(limit)],
                         cwd=ROOT, capture_output=True)
    rtxt = ref.stdout.decode("utf-8", "replace").strip().splitlines()
    if not rtxt:
        print("  ppocr_ref.py produced nothing:", ref.stderr.decode("utf-8", "replace")[:300])
        return
    b = json.loads(rtxt[-1])

    print("  det size  cpp %s  ref %s  %s" % (a["det"], b["det"],
                                              "OK" if a["det"] == b["det"] else "MISMATCH"))
    print("  lines     cpp %d  ref %d" % (len(a["lines"]), len(b["lines"])))

    used = set()
    same_text = 0
    diffs = []
    devs = []
    for la in a["lines"]:
        best, bi = 0.0, -1
        for i, lb in enumerate(b["lines"]):
            if i in used:
                continue
            v = quad_iou(la["quad"], lb["quad"])
            if v > best:
                best, bi = v, i
        if bi < 0 or best < 0.5:
            diffs.append(("unmatched", la["text"], "", best))
            continue
        used.add(bi)
        devs.append(float(np.abs(np.array(la["quad"]) - np.array(b["lines"][bi]["quad"])).max()))
        if la["text"] == b["lines"][bi]["text"]:
            same_text += 1
        else:
            diffs.append(("text", la["text"], b["lines"][bi]["text"], best))
    unmatched_ref = [b["lines"][i]["text"] for i in range(len(b["lines"])) if i not in used]
    matched = len(used)
    print("  matched   %d boxes (IoU>=0.5), identical text on %d of them" % (matched, same_text))
    if devs:
        d = np.array(devs)
        print("  corners   worst-corner offset vs reference: median %.1f px, p90 %.1f, max %.1f"
              % (np.median(d), np.percentile(d, 90), d.max()))
    if diffs or unmatched_ref:
        sys.stdout.reconfigure(encoding="utf-8")
        for kind, x, y, iou in diffs:
            print("    %-9s cpp %-28s ref %-28s IoU %.2f" % (kind, x, y, iou))
        for t in unmatched_ref:
            print("    %-9s cpp %-28s ref %-28s" % ("ref-only", "", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=os.path.join("assets", "japan_2.jpg"))
    ap.add_argument("--line", default=None, help="a cropped text line, for the rec tensor check")
    ap.add_argument("--limit", type=int, default=960)
    ap.add_argument("--skip-pipeline", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(EXE):
        print("build the CLI first:  sh build/gcc.sh pure/ppocr.cpp -o ppocr.exe")
        return 1

    print("TENSOR (C++ interpreter vs onnxruntime, same input tensor)")
    worst = []
    worst.append(tensor_level(args.img, "det", os.path.join(ROOT, "models",
                                                            "ppocrv5-mobile-det.onnx"), args.limit))
    line = args.line or args.img
    worst.append(tensor_level(line, "rec", os.path.join(ROOT, "models",
                                                        "ppocrv5-mobile-rec.onnx"), args.limit))
    print()
    if not args.skip_pipeline:
        print("PIPELINE (ppocr run vs tools/ppocr_ref.py)")
        pipeline_level(args.img, args.limit)

    bad = [w for w in worst if w is None or w > 1e-3]
    print()
    print("tensor parity: %s" % ("FAIL" if bad else "OK (relative max error < 1e-3)"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
