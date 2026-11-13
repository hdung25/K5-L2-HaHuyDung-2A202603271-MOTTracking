"""Bước 2 — Từ điểm mồi của bảng nhận diện đề A, nối detection thành quỹ đạo dày (dense) cho P1–P5.

Thuật toán nối (single-object, hai chiều từ frame mồi):
  * Dự đoán box frame kế tiếp bằng vận tốc không đổi ước lượng trên 5 frame gần nhất
    (cùng giả định với nội suy tuyến tính của CVAT).
  * Ghép với detection có điểm  s = IoU(pred, det) + 0.15·[cùng ID BoT-SORT] ;
    nhận nếu IoU ≥ 0.35. Không có thì đánh dấu "mất" và tiếp tục dự đoán (tối đa 20 frame).
  * Một detection không được dùng cho hai người trong cùng frame (giải xung đột theo IoU).

Đầu ra: out/dense_raw.json  {Pk: {frame: [x1,y1,x2,y2,conf] | null}}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
CFG = json.loads((ROOT / "src" / "lab_variants.json").read_text(encoding="utf-8"))
VARIANT = "A"
N = 150
# P6 = nam áo khoác đen, quần kaki, đeo balo đi ngang sát camera; P7 = mẹ của bé gái P5
EXTRA = {"P6": (34, [225, 560]), "P7": (0, [1326, 520])}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def load():
    d = json.loads((OUT / "detections.json").read_text())
    raw = {int(k): v for k, v in d["raw"].items()}
    trk = {int(k): v for k, v in d["tracked"].items()}
    # gắn ID tracker vào detection thô nếu trùng box (IoU > 0.9)
    for f in range(N):
        for r in raw[f]:
            best = max(trk.get(f, []), key=lambda t: iou(r, t), default=None)
            if best is not None and iou(r, best) > 0.9:
                r[5] = best[5]
    return raw


def seed_det(dets, point):
    x, y = point
    cands = [d for d in dets if d[0] <= x <= d[2] and d[1] <= y <= d[3]]
    if not cands:
        raise SystemExit(f"Không có detection chứa điểm mồi {point}")
    # ưu tiên box nhỏ nhất (người ở gần điểm nhất), conf cao
    return min(cands, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]) / max(d[4], 0.05))


def predict(hist: dict[int, list], f: int, step: int):
    known = sorted(k for k, v in hist.items() if v is not None)
    known = [k for k in known if (k - f) * step < 0]  # các frame phía "đã đi qua"
    known = sorted(known, key=lambda k: abs(k - f))[:5]
    if not known:
        return None
    if len(known) == 1:
        return hist[known[0]][:4]
    ks = np.array(known, float)
    B = np.array([hist[k][:4] for k in known], float)
    A = np.vstack([ks, np.ones_like(ks)]).T
    coef, *_ = np.linalg.lstsq(A, B, rcond=None)
    return (coef[0] * f + coef[1]).tolist()


def link(raw, seed_frame, point, used):
    hist: dict[int, list | None] = {}
    s = seed_det(raw[seed_frame], point)
    hist[seed_frame] = s
    used.setdefault(seed_frame, []).append(s)
    for step in (1, -1):
        miss = 0
        last_tid = s[5]
        f = seed_frame + step
        while 0 <= f < N and miss <= 30:
            p = predict(hist, f, step)
            # đang mất dấu: thử thêm giả thiết "đứng yên tại vị trí cuối" (người bị che thường
            # hiện lại gần chỗ mất), tránh vận tốc ngoại suy kéo box đi quá xa.
            last = next(hist[k] for k in range(f - step, -1 if step > 0 else N, -step) if hist.get(k))
            preds = [p] if miss == 0 else [p, last[:4]]
            best, score = None, 0.0
            for d in raw[f]:
                if any(d is u for u in used.get(f, [])):
                    continue
                v = max(iou(q, d) for q in preds)
                sc = v + (0.15 if d[5] == last_tid and d[5] != -1 else 0)
                if v >= 0.35 and sc > score:
                    best, score = d, sc
            hist[f] = best
            if best is None:
                miss += 1
            else:
                miss = 0
                last_tid = best[5] if best[5] != -1 else last_tid
                used.setdefault(f, []).append(best)
            f += step
    return {f: hist.get(f) for f in range(N)}


def main():
    raw = load()
    used: dict[int, list] = {}
    dense = {}
    for k, mot_id in enumerate(CFG["variants"][VARIANT], start=1):
        t = CFG["targets"][str(mot_id)]
        tr = link(raw, t["image"] - 1, t["point"], used)
        dense[f"P{k}"] = {f: (v[:5] if v else None) for f, v in tr.items()}
        have = [f for f, v in tr.items() if v]
        print(f"P{k} (MOT {mot_id}): {len(have)} frame có detection, {min(have)}–{max(have)}")
    # Người thêm theo yêu cầu (ngoài 5 người của đề): điểm mồi = (frame, điểm trên người)
    for pid, (f0, pt) in EXTRA.items():
        tr = link(raw, f0, pt, used)
        dense[pid] = {f: (v[:5] if v else None) for f, v in tr.items()}
        have = [f for f, v in tr.items() if v]
        print(f"{pid} (thêm): {len(have)} frame có detection, {min(have)}–{max(have)}")
    (OUT / "dense_raw.json").write_text(json.dumps(dense))


if __name__ == "__main__":
    main()
