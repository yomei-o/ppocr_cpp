"""Render a Japanese test page — horizontal lines, vertical (tategaki) columns, and small text.

The point is coverage the downloaded sample does not give: the sample is a word cloud of short
horizontal phrases, so it never exercises the vertical-text path (a crop taller than 1.5x its width
gets rotated 90 degrees before recognition) or long sentences.

  python tools/make_test_image.py --out assets/page_ja.png
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

FONTS = [
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    r"C:\Windows\Fonts\NotoSansJP-VF.ttf",
]

HORIZ = [
    (36, "第三章 走行中の異常と対処"),
    (26, "エンジンの回転が不安定なときは、まず燃料フィルタの詰まりを確認してください。"),
    (26, "警告灯が点灯した場合は安全な場所に停車し、取扱説明書の 128 ページを参照。"),
    (22, "型式 GX-4200／製造番号 SN-2024-018837／定格出力 3.6kW（50Hz）"),
    (18, "※ 純正部品以外を使用した場合、保証の対象外となることがあります。"),
    (30, "点検は 3,000km ごと、または 6 ヶ月ごとに実施"),
]
VERT = ["春はあけぼの", "やうやう白くなりゆく", "山ぎは少し明かりて"]


def pick_font(size):
    for path in FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("assets", "page_ja.png"))
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    img = Image.new("RGB", (args.width, args.height), (250, 249, 246))
    d = ImageDraw.Draw(img)

    y = 48
    for size, text in HORIZ:
        d.text((60, y), text, font=pick_font(size), fill=(24, 24, 28))
        y += int(size * 2.1)

    # tategaki: one glyph per row, right to left
    x = args.width - 90
    for col in VERT:
        f = pick_font(34)
        cy = 520
        for ch in col:
            d.text((x, cy), ch, font=f, fill=(24, 24, 28))
            cy += 42
        x -= 64

    # a boxed caption, and a low-contrast line the detector may or may not find
    d.rectangle([60, 700, 520, 760], outline=(120, 120, 130), width=2)
    d.text((78, 716), "図 4-2 冷却水の経路", font=pick_font(28), fill=(24, 24, 28))
    d.text((60, 800), "薄い文字のテスト（コントラスト低）", font=pick_font(24), fill=(190, 190, 196))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    img.save(args.out)
    print("wrote", args.out, img.size)


if __name__ == "__main__":
    main()
