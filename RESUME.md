# RESUME — 現状と次の一手

最終更新: 2026-08-24（初版が動いた時点）

## いまどこまで動くか

PP-OCRv5 mobile の検出 + 認識が、**onnxruntime なしのスクラッチ C++** で通っている。
native（MSVC / mingw）と WASM（Emscripten / node / ブラウザ）の両方。

- `ppocr run --img assets/japan_2.jpg` → 50 行を検出・認識。
- onnxruntime とのテンソル一致: 検出 **1.0e-04**、認識 **3.4e-05**（相対最大）。
- PaddleOCR 本来の実装（cv2 + pyclipper）とのパイプライン一致: 枠 49/50 対応、
  テキスト 47/49 一致、角のずれ中央値 1 px。
- WASM と native は `japan_2.jpg` の 50 行すべてで同一文字列。
- ブラウザデモ `wasm/index.html` は検出→枠表示→行ごとに認識して埋める。

数字と限界は README に全部書いてある。ここは**次に触るときの手掛かり**だけ。

## ビルド（この組み合わせが最速）

```sh
EXTRA="-DUSE_EIGEN -mavx2 -mfma -DPURE_SERIAL" sh build/gcc.sh pure/ppocr.cpp -o ppocr.exe
sh build/emcc.sh wasm/ppocr_wasm.cpp -o wasm/ppocr.js
```

`-DPURE_SERIAL` を**外すと遅くなる**。理由は README の速度節。OpenMP を付けるのも遅い。
これは「並列化を諦めた」のではなく、op が小さくて数が多いという形状の帰結。
本気で並列化するなら **永続スレッドプール**（呼び出しごとの生成をやめる）+
`EIGEN_DONT_PARALLELIZE` の組み合わせを試すこと。呼び出しごとの `std::thread` 生成が主犯。

## ファイルの役割

| ファイル | 役割 | 触るときの注意 |
|---|---|---|
| `pure/onnx.hpp` | protobuf コーデック + Graph IR | 姉妹リポジトリと共通。**変更しない**（差分が読めなくなる） |
| `pure/onnx_run.hpp` | インタプリタ。int64 経路 / 生存区間解放 / Weights キャッシュ / conv・転置conv・pooling | 新しい op はここ。未対応 op は `exit(1)` で名前を出す |
| `pure/ew.hpp` | 要素演算（ループ境界ホイスト済み） | 継承 engine の同名 op は使わない。理由はヘッダ冒頭 |
| `pure/fastmath.hpp` | `fm::exp_` | mingw の `std::exp` が 152 ns/call なので存在する |
| `pure/img.hpp` | 前処理・射影切り出し・描画 | **BGR 固定**。RGB にすると静かに精度が落ちる |
| `pure/dbnet.hpp` | DB 後処理 | unclip は矩形近似。`min_area_rect(poly, d)` の `grow` は目的関数の一部 |
| `pure/ctc.hpp` | 文字テーブル + greedy | `dict 行数 + 2 == head 幅` を検査して落とす |
| `pure/pipeline.hpp` | det → crop → rec | |
| `pure/ppocr.cpp` | CLI（run / det / rec / info / bench / dump / rgba） | |
| `wasm/ppocr_wasm.cpp` | 検出と認識を**別関数で**公開 | 1 回の run で返すと 10 秒間なにも出ない |
| `tools/ppocr_ref.py` | 参照実装（onnxruntime + cv2 + pyclipper） | C++ の移植ではない。だから比較に意味がある |
| `tools/parity.py` | テンソル段 + パイプライン段の比較 | **op を足したら必ず走らせる** |

## 開発中に踏んだ罠（同じ穴に落ちないように）

1. **`Conv` の stride が軸ごとに違う**。認識器は `(2,1)` と `(1,2)` で高さだけ縮める。
   継承 engine の `conv2d` は stride 1 個しか取らないので `conv2d_gen` を書いた。
2. **1-D テンソルはチャネル方向のブロードキャストではない**。ONNX は右詰めなので `[C]` は W 方向。
   これらのグラフは必ず `[1,C,1,1]` に Reshape してから足すので、そちらだけ速いパスに乗せる。
3. **重みが initializer ではなく Constant ノード**（paddle2onnx）。initializer は 0 個。
   `onx::Weights` で 1 回だけ実体化する。
4. **`sort_boxes` を `std::sort` 1 回で書くと壊れる**。「y が 10 px 以内なら x で比較」は
   推移的でない → UB。PaddleOCR と同じ 2 パス（ソート + 局所バブル）にした。
5. **`--verbose` でプロファイルしてはいけない**。ノードごとの `fflush` が op より重く、
   安いノードが一番高く見える。`ppocr bench` を使う（`onx::Prof` は印字しない）。
6. **マシンが混んでいると計測が 4 倍ぶれる**。順位だけ信じて、絶対値は空いてから測り直す。

## 次にやるなら（優先度順）

1. **縦書き**。いまは列が細切れになる（参照実装も同じ）。検出側の問題なので、
   PP-OCR の det を差し替えるか、細切れの枠を「同一列」として縦にマージする後処理を足す。
   後者はこのリポジトリだけで完結するので先に試す価値がある。
2. **文字方向分類器（cls）**。`ppocrv5-cls.onnx`（0.6 MB）が同じ配布元にある。
   180° 回転した行が読めるようになる。op は既に足りているはず。
3. **永続スレッドプール**（上記）。native の 5 s が 1〜2 s になる余地がある。
4. **server モデル**（det 88 MB / rec 84 MB）を選べるようにする。精度は上がるが WASM には重い。
   native CLI だけの選択肢として `--det/--rec` で既に渡せる（未検証）。
5. **中間テンソルのゼロ埋め回避**。`make_tensor` が data をゼロ埋めするが、要素演算は全部
   上書きする。検出 1 回で ~100 ms 程度の無駄。`Node::data` が `std::vector<float>` なので
   素直には避けられない（アロケータか、上書き前提の別コンテナが必要）。
6. **Python と C++ の機能パリティ**をもう一段。いま Python 側は参照実装だけで、
   `ppocr det` / `bench` / `dump` に相当するサブコマンドがない。

## 検証の回し方

```sh
python tools/parity.py --img assets/japan_2.jpg --line assets/line_ja.png   # op を足したら必須
python tools/parity.py --img assets/page_ja.png --line assets/line_ja.png
node wasm/test_node.js                                                    # WASM と native の一致
```

`assets/line_ja.png` は `assets/japan_2.jpg` の 1 行を切ったもの（`(115,216)-(567,286)`）。
無ければ作り直す:

```sh
python -c "from PIL import Image; Image.open('assets/japan_2.jpg').convert('RGB').crop((115,216,567,286)).save('assets/line_ja.png')"
```
