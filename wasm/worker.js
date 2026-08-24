// Inference worker. Detection is seconds and each text line is a few hundred milliseconds more, so
// none of this can run on the page's thread. The worker reports each line as it finishes rather
// than one result at the end — on a dense page that is the difference between a demo that looks
// broken and one that visibly works.
//
// Messages in:  {type:'init', det, rec, dict}                     ArrayBuffers + dict text
//               {type:'run', pixels, w, h, limit, thresh, boxThresh, unclip, drop}
//               {type:'crop', index}
// Messages out: {type:'ready', det, rec, classes} | {type:'log', text}
//               {type:'boxes', json, ms} | {type:'line', index, text, conf}
//               {type:'done', json, ms} | {type:'crop', index, rgb, w, h} | {type:'error', text}
let M = null;
let ready = false;

const log = (text) => self.postMessage({ type: 'log', text });

function copyIn(bytes) {
  const p = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, p);
  return p;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      self.importScripts('ppocr.js');
      M = await createPpocr();

      const n = M.lengthBytesUTF8(m.dict) + 1;
      const dp = M._malloc(n);
      M.stringToUTF8(m.dict, dp, n);
      const classes = M._pp_load_dict(dp);
      M._free(dp);
      log('文字テーブル ' + classes + ' クラス');

      const loadModel = (buf, fn, name) => {
        const b = new Uint8Array(buf);
        const p = copyIn(b);
        const r = fn(p, b.length);
        M._free(p);
        log(name + (r > 0 ? ' ok (' + r + ' ノード, ' + (b.length / 1048576).toFixed(1) + ' MB)'
                          : ' 読み込み失敗'));
        return r;
      };
      const det = loadModel(m.det, M._pp_load_det, '検出モデル');
      const rec = loadModel(m.rec, M._pp_load_rec, '認識モデル');
      ready = det > 0 && rec > 0 && classes > 2;
      self.postMessage({ type: 'ready', det, rec, classes });
      return;
    }

    if (m.type === 'crop') {
      const cap = 4096 * 256 * 3;
      const p = M._malloc(cap);
      const packed = M._pp_crop(m.index, p, cap);
      if (packed > 0) {
        const w = packed >>> 16, h = packed & 0xffff;
        const rgb = M.HEAPU8.slice(p, p + w * h * 3);
        self.postMessage({ type: 'crop', index: m.index, rgb, w, h }, [rgb.buffer]);
      }
      M._free(p);
      return;
    }

    if (m.type === 'run') {
      if (!ready) return;
      const px = new Uint8Array(m.pixels);
      const p = copyIn(px);
      const t0 = performance.now();
      const count = M._pp_detect(p, m.w, m.h, m.limit, m.thresh, m.boxThresh, m.unclip);
      M._free(p);
      if (count < 0) { self.postMessage({ type: 'error', text: '検出モデルが未ロード' }); return; }
      const boxJson = M.UTF8ToString(M._pp_boxes_json());
      self.postMessage({ type: 'boxes', json: boxJson, ms: performance.now() - t0 });

      for (let i = 0; i < count; ++i) {
        const conf = M._pp_rec_line(i, m.drop);
        const text = M.UTF8ToString(M._pp_line_text());
        self.postMessage({ type: 'line', index: i, text, conf });
      }
      self.postMessage({ type: 'done', json: M.UTF8ToString(M._pp_result()),
                         ms: performance.now() - t0 });
    }
  } catch (err) {
    self.postMessage({ type: 'error', text: String((err && err.message) || err) });
  }
};
