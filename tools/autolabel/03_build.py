"""Bước 3 — Quỹ đạo dày → làm mượt → chọn keyframe → file CVAT for video 1.1 (annotations.xml).

Đầu vào
  out/dense_raw.json           quỹ đạo từ detector (bước 2)
  manual.json                  sửa tay sau khi soát bằng mắt:
      {"P1": {"boxes": {"35": [x1,y1,x2,y2], ...},      # box đặt tay (ưu tiên hơn detector)
              "drop":  [[a,b], ...],                      # bỏ detection sai trong đoạn [a,b]
              "visible": [[a,b], ...],                    # đoạn có mặt (ngoài đoạn = outside)
              "occluded": [[a,b], ...],                   # đoạn bật cờ occluded
              "amodal": [[a,b], ...],                     # đoạn bị che: bỏ detector, nội suy toàn thân
              "force_kf": [f, ...]}}

Thuật toán
  1. Làm mượt nhiễu detector bằng Savitzky–Golay (cửa sổ 9, bậc 2) trên từng đoạn liên tục,
     tham số hoá box theo (cx, cy, w, h).
  2. Đoạn thiếu detection / đoạn amodal: nội suy tuyến tính giữa hai mép đoạn (đúng như CVAT).
  3. Chọn keyframe kiểu Ramer–Douglas–Peucker theo IoU: bắt đầu bằng keyframe bắt buộc
     (đầu/cuối đoạn có mặt, mép đoạn occluded); với mỗi khoảng [k0,k1] tính box nội suy tuyến tính
     b(t) = b(k0) + u·(b(k1) − b(k0)); nếu min IoU(b(t), box thật) < THR thì thêm keyframe ở frame
     tệ nhất, lặp lại. Kết quả: số keyframe ít nhất sao cho mọi frame nội suy có IoU ≥ THR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
N = 150
W, H = 1920, 1080


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def in_ranges(f, ranges):
    return any(a <= f <= b for a, b in ranges)


def to_cxcywh(b):
    return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]]


def to_xyxy(c):
    return [c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2]


def lerp(b0, b1, u):
    return [p + u * (q - p) for p, q in zip(b0, b1)]


def dense_track(raw: dict, man: dict) -> dict[int, list]:
    visible = man["visible"]
    amodal = man.get("amodal", [])
    drop = man.get("drop", [])
    manual = {int(k): v for k, v in man.get("boxes", {}).items()}
    obs: dict[int, list] = {}
    for f in range(N):
        if not in_ranges(f, visible):
            continue
        if f in manual:
            obs[f] = ("m", manual[f])
        elif raw.get(f) and not in_ranges(f, amodal) and not in_ranges(f, drop):
            obs[f] = ("d", raw[f][:4])
    # 1) làm mượt detector trên từng đoạn liên tục (box tay giữ nguyên)
    frames = sorted(obs)
    runs, cur = [], []
    for f in frames:
        if cur and f == cur[-1] + 1 and obs[f][0] == "d":
            cur.append(f)
        else:
            if cur:
                runs.append(cur)
            cur = [f] if obs[f][0] == "d" else []
            if obs[f][0] == "m":
                runs.append([f])
    if cur:
        runs.append(cur)
    smooth: dict[int, list] = {}
    for r in runs:
        arr = np.array([to_cxcywh(obs[f][1]) for f in r])
        if len(r) >= 9 and obs[r[0]][0] == "d":
            arr = savgol_filter(arr, 9, 2, axis=0, mode="interp")
        for f, c in zip(r, arr):
            smooth[f] = to_xyxy(list(map(float, c)))
    # 2) lấp chỗ trống trong đoạn có mặt bằng nội suy tuyến tính
    # (nội suy xuyên qua cả đoạn outside: box amodal của người bị che đi tiếp theo quỹ đạo cũ)
    out: dict[int, list] = {}
    known = sorted(smooth)
    for a, b in visible:
        for f in range(a, b + 1):
            if f in smooth:
                out[f] = smooth[f]
                continue
            prv = max((k for k in known if k < f), default=None)
            nxt = min((k for k in known if k > f), default=None)
            if prv is None:
                out[f] = smooth[nxt]
            elif nxt is None:
                out[f] = smooth[prv]
            else:
                out[f] = lerp(smooth[prv], smooth[nxt], (f - prv) / (nxt - prv))
    # sửa mép box theo mốc đo tay (vd. đáy box = bàn chân thấp nhất khi detector cắt hụt chân)
    for edge, idx in (("x1", 0), ("y1", 1), ("x2", 2), ("y2", 3)):
        pts = sorted((int(k), v) for k, v in man.get(edge, {}).items())
        if pts:
            fs, vs = [q[0] for q in pts], [q[1] for q in pts]
            for f in out:
                if fs[0] <= f <= fs[-1]:
                    out[f][idx] = float(np.interp(f, fs, vs))
    # cắt vào khung ảnh
    for f, b in out.items():
        out[f] = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
    return out


def select_keyframes(dense: dict[int, list], forced: set[int], thr: float, visible) -> list[int]:
    kfs = set(forced)
    for a, b in visible:
        kfs |= {a, b}
    changed = True
    while changed:
        changed = False
        ks = sorted(kfs)
        for k0, k1 in zip(ks, ks[1:]):
            if not (in_ranges(k0, visible) and in_ranges(k1, visible)):
                continue
            if any(not in_ranges(f, visible) for f in range(k0, k1 + 1)):
                continue
            worst, wi = 1.0, None
            for t in range(k0 + 1, k1):
                v = iou(lerp(dense[k0], dense[k1], (t - k0) / (k1 - k0)), dense[t])
                if v < worst:
                    worst, wi = v, t
            if wi is not None and worst < thr:
                kfs.add(wi)
                changed = True
    return sorted(kfs)


def interp_track(dense, kfs, visible):
    """Box CVAT sẽ hiển thị khi chỉ có keyframe (để đo chất lượng)."""
    res = {}
    ks = sorted(kfs)
    for f in range(N):
        if not in_ranges(f, visible):
            continue
        k0 = max(k for k in ks if k <= f)
        k1 = min((k for k in ks if k >= f), default=k0)
        res[f] = dense[k0] if k1 == k0 else lerp(dense[k0], dense[k1], (f - k0) / (k1 - k0))
    return res


def build(thr: float):
    raw = json.loads((OUT / "dense_raw.json").read_text())
    manual = json.loads((HERE / "manual.json").read_text(encoding="utf-8"))
    final, summary, tracks = {}, {}, []
    for pid in sorted(manual):
        man = manual[pid]
        r = {int(k): v for k, v in raw[pid].items()}
        dense = dense_track(r, man)
        occ = man.get("occluded", [])
        forced = set(man.get("force_kf", []))
        for a, b in occ:
            forced |= {a, b + 1}              # bật occluded ở a, tắt ở b+1
        forced = {f for f in forced if in_ranges(f, man["visible"])}
        kfs = select_keyframes(dense, forced, thr, man["visible"])
        shown = interp_track(dense, kfs, man["visible"])
        ious = [iou(shown[f], dense[f]) for f in shown]
        summary[pid] = {"kf": kfs, "n_kf": len(kfs), "n_vis": len(shown),
                        "min_iou": round(min(ious), 3), "mean_iou": round(float(np.mean(ious)), 3)}
        final[pid] = {f: [*shown[f], 0, pid + ("*" if f in kfs else "")] for f in shown}
        final[pid + "_dense"] = {f: [*dense[f], 0, ""] for f in dense}
        tracks.append((pid, dense, kfs, man["visible"], occ))
    (OUT / "final.json").write_text(json.dumps({**{k: v for k, v in final.items() if not k.endswith('_dense')}}))
    (OUT / "dense.json").write_text(json.dumps({k[:-6]: v for k, v in final.items() if k.endswith('_dense')}))
    return tracks, summary


def xml_tracks(tracks) -> str:
    """Thẻ <track> theo CVAT for video 1.1 (chỉ keyframe + mốc outside)."""
    lines = []
    for tid, (pid, dense, kfs, visible, occ) in enumerate(tracks):
        lines.append(f'  <track id="{tid}" label="person" source="manual">')
        boxes = []
        for f in kfs:
            boxes.append((f, dense[f], 0, int(in_ranges(f, occ))))
        # mốc outside: frame ngay sau mỗi đoạn có mặt (nếu còn trong job)
        for a, b in visible:
            if b + 1 < N:
                boxes.append((b + 1, dense[b], 1, 0))
        for f, bx, outside, oc in sorted(boxes):
            lines.append(
                f'    <box frame="{f}" keyframe="1" outside="{outside}" occluded="{oc}" '
                f'xtl="{bx[0]:.2f}" ytl="{bx[1]:.2f}" xbr="{bx[2]:.2f}" ybr="{bx[3]:.2f}" z_order="0">'
                f"</box>"
            )
        lines.append("  </track>")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.85)
    a = ap.parse_args()
    tracks, summary = build(a.thr)
    for pid, s in summary.items():
        print(f"{pid}: {s['n_kf']:>2} kf / {s['n_vis']} frame  min IoU {s['min_iou']}  mean {s['mean_iou']}  kf={s['kf']}")
    (OUT / "tracks.xml").write_text(xml_tracks(tracks), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", OUT / "tracks.xml")


if __name__ == "__main__":
    main()
