# Third-party notices

This repository's own code (`pure/`, `wasm/*.cpp`, `wasm/*.js`, `wasm/index.html`, `tools/`,
`build/`) is BSD 3-Clause — see `LICENSE`. The files below are third party and keep their own
licenses.

## Model weights

`models/ppocrv5-mobile-det.onnx`, `models/ppocrv5-mobile-rec.onnx`, `models/ppocrv5_dict.txt`

PP-OCRv5 mobile detection and recognition, from **PaddleOCR** — Apache License 2.0.
https://github.com/PaddlePaddle/PaddleOCR

The ONNX conversions were obtained from https://huggingface.co/bukuroo/PPOCRv5-ONNX
(the same Apache-2.0 weights, exported with paddle2onnx). The character table is PaddleOCR's
`ppocrv5_dict.txt` verbatim.

## Test images

`assets/japan_2.jpg` — PaddleOCR's Japanese sample image (`doc/imgs/japan_2.jpg`), Apache-2.0.
`assets/japan_demo_pair.jpg` — PaddleOCR's side-by-side demo figure for the same image, Apache-2.0.
`assets/page_ja.png` — generated locally by `tools/make_test_image.py`, rendered with fonts
installed on the build machine. Not redistributed content of its own beyond the glyph shapes it
rasterizes; regenerate it rather than treating it as an asset with provenance.

## Vendored source

`pure/third_party/stb_image.h`, `pure/third_party/stb_image_write.h`
Sean Barrett's stb — public domain / MIT (dual). https://github.com/nothings/stb

`pure/third_party/eigen_flat/`
Eigen 3 (Core only, flattened into single-directory includes) — MPL2. See `COPYING.MPL2` and
`COPYING.README` in that directory. https://eigen.tuxfamily.org

## Engine lineage

`pure/onnx.hpp`, `pure/autograd.hpp`, `pure/nd.hpp`, `pure/ops2d.hpp`, `pure/face_ops.hpp`,
`pure/linalg.hpp`, `pure/bn.hpp`, `pure/ops_yolox.hpp`, `pure/backend.hpp`, `pure/parallel.hpp`
come from this author's own sibling repositories (`cudnn_cpp`, `yolov8_cpp`, `yolo_lpr_cpp`) and are
BSD 3-Clause like the rest of this repository. Two of them carry a local change:
`pure/parallel.hpp` adds the `PURE_SERIAL` build switch, and `pure/autograd.hpp` adds
`infer_only()` so `make_tensor` can skip the grad buffer. The rest are unmodified, so the sibling
repos and this one stay diffable.

## Python-side reference only

`tools/ppocr_ref.py` and `tools/parity.py` import onnxruntime (MIT), OpenCV (Apache-2.0),
pyclipper (BSL-1.0), Shapely (BSD-3) and NumPy (BSD-3). `tools/wasm_smoke.py` uses Playwright
(Apache-2.0) and `tools/make_test_image.py` uses Pillow (MIT-CMU). None of these are needed to
build or run the C++ or WASM engine — they exist so the scratch implementation can be checked
against the reference one, and so the demo page can be driven in a real browser.
