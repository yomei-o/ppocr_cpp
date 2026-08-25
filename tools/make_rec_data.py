"""Render a small recogniser training set: cropped text lines plus a PaddleOCR-style label list.

  python tools/make_rec_data.py --out scratch/recdata --n 400 --val 100

WHY SYNTHESISE. Fine-tuning needs image/label pairs that agree, and hand-labelled data does not
agree often enough to debug a training loop with: when the loss will not fall you cannot tell
whether the gradient is wrong or the label is. Rendering the text means the label is the text that
was drawn, by construction.

WHAT IT IS FOR. This is a harness, not a dataset. Every line is one font at one size on white, so a
model that learns it has learned nothing transferable — the point is to show the loop moves the
validation accuracy, which is what proves forward, backward and optimiser agree on a real model.

THE CHARACTER SET IS FILTERED AGAINST THE DICTIONARY. A label containing a character PP-OCRv5 cannot
emit is not a hard sample, it is an impossible one, and it teaches the model to suppress a character
it should be producing. tools/../pure/train_rec.hpp refuses such a label; this generator never makes
one.
"""
import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFont

FONTS = [
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    r"C:\Windows\Fonts\NotoSansJP-VF.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

# Vocabulary the lines are built from. Deliberately narrow: a few hundred samples cannot teach a
# 18385-class head anything broad, and a narrow set makes the accuracy move visibly.
WORDS = [
    "型式", "製造番号", "定格出力", "電圧", "電流", "周波数", "質量", "全長", "全幅", "全高",
    "始動方式", "燃料", "潤滑油", "冷却方式", "点火", "排気量", "回転数", "騒音", "保証", "認証",
    "取扱説明書", "安全上のご注意", "定期点検", "部品番号", "製造年月",
]
UNITS = ["kW", "V", "A", "Hz", "kg", "mm", "L", "rpm", "dB", "MPa"]


def pick_font(size):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    raise SystemExit("no CJK font found; edit FONTS at the top of this script")


def load_dict_chars(dict_path):
    """The set of strings PP-OCRv5 can emit, so nothing impossible reaches a label."""
    ok = {" "}
    with open(dict_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                ok.add(line)
    return ok


def make_text(rng, allowed):
    kind = rng.random()
    if kind < 0.45:
        s = "%s %s-%04d" % (rng.choice(WORDS), rng.choice("ABCDEFGHJKLMNPRSTUVWXY"),
                            rng.randrange(10000))
    elif kind < 0.75:
        s = "%s %d.%d%s" % (rng.choice(WORDS), rng.randrange(1, 400), rng.randrange(10),
                            rng.choice(UNITS))
    else:
        s = "%s／%s" % (rng.choice(WORDS), rng.choice(WORDS))
    return "".join(c for c in s if c in allowed)


def render(text, size, pad=6):
    font = pick_font(size)
    probe = Image.new("RGB", (8, 8), "white")
    box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    img = Image.new("RGB", (w + pad * 2, h + pad * 2), "white")
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), text, font=font, fill="black")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratch/recdata")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--dict", default="models/ppocrv5_dict.txt")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    allowed = load_dict_chars(args.dict)
    rng = random.Random(args.seed)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    # Distinct texts only. Sharing a line between train and val would report memorisation as
    # accuracy, which is the exact mistake the sibling repo documents for its synthetic set.
    texts, seen = [], set()
    guard = 0
    while len(texts) < args.n + args.val and guard < (args.n + args.val) * 200:
        guard += 1
        t = make_text(rng, allowed)
        if len(t) < 3 or t in seen:
            continue
        seen.add(t)
        texts.append(t)
    if len(texts) < args.n + args.val:
        raise SystemExit("could only build %d distinct lines" % len(texts))

    rows = []
    for i, t in enumerate(texts):
        img = render(t, rng.choice([22, 26, 30, 34]))
        rel = "images/%05d.png" % i
        img.save(os.path.join(args.out, rel))
        rows.append("%s\t%s" % (rel, t))

    for name, part in (("train.txt", rows[:args.n]), ("val.txt", rows[args.n:])):
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(part) + "\n")
        print("%s: %d lines" % (os.path.join(args.out, name), len(part)))


if __name__ == "__main__":
    main()
