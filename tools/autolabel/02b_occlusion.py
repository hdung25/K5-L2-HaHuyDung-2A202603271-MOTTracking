"""Ước lượng tỉ lệ bị che (giống cách MOT17 tính cột visibility): phần box mục tiêu nằm
dưới hợp các box người đứng GẦN camera hơn (đáy box thấp hơn trong ảnh).
Box mục tiêu ở frame thiếu detection = nội suy tuyến tính giữa hai detection gần nhất."""
import json
import numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"
raw = {int(k): v for k, v in json.loads((OUT / "detections.json").read_text())["raw"].items()}
dense = json.loads((OUT / "dense_raw.json").read_text())

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy
    return ix*iy/u if u else 0

res = {}
for p, tr in dense.items():
    tr = {int(k): v for k, v in tr.items()}
    known = sorted(k for k, v in tr.items() if v)
    cov = {}
    for f in range(150):
        b = tr.get(f)
        if not b:
            prv = max((k for k in known if k < f), default=None); nxt = min((k for k in known if k > f), default=None)
            if prv is None or nxt is None: continue
            u = (f-prv)/(nxt-prv); b = [tr[prv][i] + u*(tr[nxt][i]-tr[prv][i]) for i in range(4)]
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        m = np.zeros((max(1, y2-y1), max(1, x2-x1)), bool)
        for d in raw[f]:
            if d[4] < 0.4 or iou(d, b) > 0.6 or d[3] < b[3] + 15: continue
            ox1, oy1, ox2, oy2 = [int(round(v)) for v in d[:4]]
            m[max(0, oy1-y1):max(0, oy2-y1), max(0, ox1-x1):max(0, ox2-x1)] = True
        cov[f] = round(float(m.mean()), 2)
    res[p] = cov
    line = " ".join(f"{f}:{c:.2f}" for f, c in cov.items() if c >= 0.15)
    print(p, line)
(OUT / "coverage.json").write_text(json.dumps(res))
