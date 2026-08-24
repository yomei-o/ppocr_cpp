// Headless smoke test for the WASM build — the same models, the same pixels, no browser.
//
//   sh build/gcc.sh pure/ppocr.cpp -o ppocr.exe
//   ./ppocr.exe rgba --img assets/japan_2.jpg --out scratch/sample.rgba
//   sh build/emcc.sh wasm/ppocr_wasm.cpp -o wasm/ppocr.js
//   node wasm/test_node.js [scratch/sample.rgba]
//
// It prints the same lines `ppocr run` prints. If the two disagree, the WASM build and the native
// build are not running the same code, which is the whole point of checking.
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const rgbaPath = process.argv[2] || path.join(ROOT, 'scratch', 'sample.rgba');

(async () => {
  const createPpocr = require('./ppocr.js');
  const M = await createPpocr();

  const copyIn = (buf) => {
    const p = M._malloc(buf.length);
    M.HEAPU8.set(buf, p);
    return p;
  };

  const dictText = fs.readFileSync(path.join(ROOT, 'models', 'ppocrv5_dict.txt'), 'utf8');
  const n = M.lengthBytesUTF8(dictText) + 1;
  const dp = M._malloc(n);
  M.stringToUTF8(dictText, dp, n);
  console.log('dict classes:', M._pp_load_dict(dp));
  M._free(dp);

  for (const [file, fn, label] of [
    ['ppocrv5-mobile-det.onnx', M._pp_load_det, 'det'],
    ['ppocrv5-mobile-rec.onnx', M._pp_load_rec, 'rec'],
  ]) {
    const buf = fs.readFileSync(path.join(ROOT, 'models', file));
    const p = copyIn(buf);
    console.log(label, 'nodes:', fn(p, buf.length));
    M._free(p);
  }

  const raw = fs.readFileSync(rgbaPath);
  const w = raw.readInt32LE(0), h = raw.readInt32LE(4);
  const px = raw.subarray(8);
  console.log('frame', w + 'x' + h);
  const pp = copyIn(px);

  const limit = Number(process.env.LIMIT || 960);
  let t0 = Date.now();
  const count = M._pp_detect(pp, w, h, limit, 0.3, 0.6, 1.5);
  M._free(pp);
  const detMs = Date.now() - t0;
  console.log(count + ' boxes in ' + detMs + ' ms');

  t0 = Date.now();
  for (let i = 0; i < count; ++i) {
    const conf = M._pp_rec_line(i, 0.5);
    const text = M.UTF8ToString(M._pp_line_text());
    if (conf > 0) console.log('  %s  %s  %s', String(i).padStart(2), (conf / 100).toFixed(2), text);
  }
  console.log('rec ' + (Date.now() - t0) + ' ms total');
})().catch((e) => { console.error(e); process.exit(1); });
