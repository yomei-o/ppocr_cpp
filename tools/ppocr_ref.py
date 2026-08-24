"""Python reference for the whole pipeline — the thing pure/*.hpp is checked against.

Deliberately *not* a port of the C++ code. It runs the graphs with onnxruntime, finds contours with
cv2, and offsets the unclip polygon with pyclipper, i.e. it is PaddleOCR's own algorithm using
PaddleOCR's own libraries. The C++ side reimplements all three from scratch, so
`python tools/parity.py` comparing the two is a real cross-check and not a tautology.

  python tools/ppocr_ref.py --img assets/japan_2.jpg [--json] [--out result.png]
  python tools/ppocr_ref.py --img line.png --rec-only

Preprocessing is transcribed from PaddleOCR, formula for formula, and the channel order is BGR
(see the note at the top of pure/img.hpp). Getting that wrong leaves everything "working" while
quietly reading worse.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET = os.path.join(ROOT, "models", "ppocrv5-mobile-det.onnx")
REC = os.path.join(ROOT, "models", "ppocrv5-mobile-rec.onnx")
DICT = os.path.join(ROOT, "models", "ppocrv5_dict.txt")

DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---- preprocessing ---------------------------------------------------------------------------
def det_input(bgr, limit_side_len=960, limit_max=True):
    """DetResizeForTest + NormalizeImage + ToCHWImage. Returns (NCHW float32, resized w, h)."""
    h, w = bgr.shape[:2]
    ratio = 1.0
    if limit_max:
        if max(h, w) > limit_side_len:
            ratio = limit_side_len / max(h, w)
    else:
        if min(h, w) < limit_side_len:
            ratio = limit_side_len / min(h, w)
    rh, rw = int(h * ratio), int(w * ratio)          # truncate, then snap to a multiple of 32
    rh = max(32, int(round(rh / 32.0)) * 32)
    rw = max(32, int(round(rw / 32.0)) * 32)
    img = cv2.resize(bgr, (rw, rh))
    x = img.astype(np.float32) / 255.0
    x = (x - DET_MEAN) / DET_STD
    x = x.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, dtype=np.float32), rw, rh


def rec_input(bgr, img_h=48, pad_w=0):
    """resize_norm_img: height 48, width = ceil(48*aspect), (x/255 - 0.5)/0.5."""
    h, w = bgr.shape[:2]
    tw = max(1, int(np.ceil(img_h * w / max(1, h))))
    W = pad_w if pad_w > 0 else tw
    tw = min(tw, W)
    img = cv2.resize(bgr, (tw, img_h))
    x = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    out = np.zeros((1, 3, img_h, W), dtype=np.float32)
    out[0, :, :, :tw] = x
    return out


# ---- DB post-process (PaddleOCR DBPostProcess, box_type='quad', score_mode='fast') -----------
def get_mini_boxes(contour):
    rect = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(rect)), key=lambda p: p[0])
    idx1, idx2, idx3, idx4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        idx1, idx4 = 0, 1
    else:
        idx1, idx4 = 1, 0
    if points[3][1] > points[2][1]:
        idx2, idx3 = 2, 3
    else:
        idx2, idx3 = 3, 2
    box = [points[idx1], points[idx2], points[idx3], points[idx4]]
    return np.array(box), min(rect[1])


def box_score_fast(bitmap, box):
    h, w = bitmap.shape[:2]
    box = box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    return expanded


def boxes_from_prob(pred, src_w, src_h, thresh=0.3, box_thresh=0.6, unclip_ratio=1.5,
                    max_candidates=1000, min_size=3):
    """pred is HxW float. Returns [(quad 4x2 in source pixels, score)]."""
    H, W = pred.shape
    bitmap = (pred > thresh).astype(np.uint8)
    outs = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[0] if len(outs) == 2 else outs[1]
    boxes = []
    for contour in contours[:max_candidates]:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        _, sside = get_mini_boxes(contour)
        if sside < min_size:
            continue
        score = box_score_fast(pred, points.reshape(-1, 2))
        if score < box_thresh:
            continue
        expanded = unclip(points, unclip_ratio)
        if len(expanded) == 0:
            continue
        box = np.array(expanded[0]).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)
        if sside < min_size + 2:
            continue
        box = np.array(box, dtype=np.float64)
        box[:, 0] = np.clip(np.round(box[:, 0] / W * src_w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / H * src_h), 0, src_h)
        boxes.append((box, float(score)))
    return boxes


def sort_boxes(boxes):
    return sorted(boxes, key=lambda b: (b[0][0][1], b[0][0][0]))


def sorted_reading_order(boxes):
    """PaddleOCR sorted_boxes: top-to-bottom, then left-to-right within 10 px."""
    b = sorted(boxes, key=lambda x: (x[0][0][1], x[0][0][0]))
    for i in range(len(b) - 1):
        for j in range(i, -1, -1):
            if abs(b[j + 1][0][0][1] - b[j][0][0][1]) < 10 and b[j + 1][0][0][0] < b[j][0][0][0]:
                b[j], b[j + 1] = b[j + 1], b[j]
            else:
                break
    return b


# ---- crop ------------------------------------------------------------------------------------
def get_rotate_crop_image(img, points):
    points = np.array(points, dtype=np.float32)
    w = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    h = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    w, h = max(1, w), max(1, h)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(points, dst)
    crop = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE,
                               flags=cv2.INTER_LINEAR)
    if crop.shape[0] * 1.0 / crop.shape[1] >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


# ---- CTC -------------------------------------------------------------------------------------
def load_dict(path=DICT):
    with open(path, "rb") as f:
        lines = f.read().decode("utf-8").split("\n")
    chars = [ln.rstrip("\r") for ln in lines]
    while chars and chars[-1] == "":
        chars.pop()
    return [""] + chars + [" "]


def ctc_greedy(probs, table):
    """probs is [T, C] (already softmaxed by the graph)."""
    idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out, kept, prev = [], [], -1
    for t, k in enumerate(idx):
        if k != 0 and k != prev:
            out.append(table[k])
            kept.append(conf[t])
        prev = k
    return "".join(out), float(np.mean(kept)) if kept else 0.0


# ---- pipeline --------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--det", default=DET)
    ap.add_argument("--rec", default=REC)
    ap.add_argument("--dict", default=DICT)
    ap.add_argument("--limit", type=int, default=960)
    ap.add_argument("--min", action="store_true", help="limit_type=min instead of max")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--box-thresh", type=float, default=0.6)
    ap.add_argument("--unclip", type=float, default=1.5)
    ap.add_argument("--drop", type=float, default=0.5)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rec-only", action="store_true", help="the image IS one cropped text line")
    args = ap.parse_args()

    bgr = cv2.imdecode(np.fromfile(args.img, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print("cannot decode", args.img)
        return 1
    table = load_dict(args.dict)

    if args.rec_only:
        sess = ort.InferenceSession(args.rec, providers=["CPUExecutionProvider"])
        x = rec_input(bgr)
        y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        text, conf = ctc_greedy(y[0], table)
        print("%s  (conf %.3f, T=%d)" % (text, conf, y.shape[1]))
        return 0

    det_sess = ort.InferenceSession(args.det, providers=["CPUExecutionProvider"])
    rec_sess = ort.InferenceSession(args.rec, providers=["CPUExecutionProvider"])

    x, rw, rh = det_input(bgr, args.limit, not args.min)
    prob = det_sess.run(None, {det_sess.get_inputs()[0].name: x})[0][0, 0]
    boxes = boxes_from_prob(prob, bgr.shape[1], bgr.shape[0], args.thresh, args.box_thresh,
                            args.unclip)
    boxes = sorted_reading_order(boxes)

    lines = []
    for quad, score in boxes:
        crop = get_rotate_crop_image(bgr, quad)
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue
        xr = rec_input(crop)
        y = rec_sess.run(None, {rec_sess.get_inputs()[0].name: xr})[0]
        text, conf = ctc_greedy(y[0], table)
        if not text or conf < args.drop:
            continue
        lines.append({"text": text, "conf": round(conf, 4), "det_score": round(score, 4),
                      "quad": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in quad]})

    if args.json:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps({"det": [rw, rh], "lines": lines}, ensure_ascii=False))
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print("%dx%d -> det %dx%d : %d boxes, %d lines kept"
              % (bgr.shape[1], bgr.shape[0], rw, rh, len(boxes), len(lines)))
        for i, ln in enumerate(lines):
            print("  %2d  %.3f  %s" % (i, ln["conf"], ln["text"]))

    if args.out:
        vis = bgr.copy()
        for ln in lines:
            cv2.polylines(vis, [np.array(ln["quad"], dtype=np.int32)], True, (60, 220, 60), 2)
        cv2.imwrite(args.out, vis)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
