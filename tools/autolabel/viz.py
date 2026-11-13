"""Vẽ contact sheet để soát bằng mắt: mỗi ô = một frame, crop quanh người, box vẽ màu.

python viz.py P1 --every 3 --layers dense_raw.json:lime final.json:red [--frames 20-60]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
FR = HERE.parents[1] / ".lab-local" / "mot17-09"
OUT = HERE / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("person")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--frames", default="0-149")
    ap.add_argument("--layers", nargs="+", default=["dense_raw.json:lime"])
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--cell", type=int, default=230)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lo, hi = map(int, a.frames.split("-"))
    layers = []
    for spec in a.layers:
        fn, color = spec.split(":")
        d = json.loads((OUT / fn).read_text())[a.person]
        layers.append(({int(k): v for k, v in d.items()}, color))
    frames = list(range(lo, hi + 1, a.every))
    # vùng crop: theo box của layer đầu (hoặc layer nào có), nới 60%
    cells = []
    last = None
    for f in frames:
        b = next((L[f] for L, _ in layers if L.get(f)), None) or last
        last = b
        img = Image.open(FR / f"{f + 1:06d}.jpg").convert("RGB")
        if b is None:
            crop_box = (0, 0, 1920, 1080)
        else:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            s = max(b[3] - b[1], (b[2] - b[0]) * 1.6) * 0.85 + 40
            crop_box = (int(cx - s), int(cy - s), int(cx + s), int(cy + s))
        dr = ImageDraw.Draw(img)
        for L, color in layers:
            v = L.get(f)
            if v:
                dr.rectangle(v[:4], outline=color, width=3)
                if len(v) > 5 and v[5]:
                    dr.text((v[0] + 4, v[1] + 4), v[5], fill=color)
        cell = img.crop(crop_box).resize((a.cell, a.cell))
        ImageDraw.Draw(cell).text((4, 4), str(f), fill="yellow")
        cells.append(cell)
    rows = (len(cells) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (a.cols * a.cell, rows * a.cell), "black")
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % a.cols) * a.cell, (i // a.cols) * a.cell))
    out = OUT / (a.out or f"sheet_{a.person}.jpg")
    sheet.save(out, quality=85)
    print(out)


if __name__ == "__main__":
    main()
