"""Bước 1 — Chạy detector người (YOLO) + tracker (BoT-SORT) trên 150 frame MOT17-09.

Kết quả: tools/autolabel/out/detections.json
  {"frames": {frame_cvat: [[x1,y1,x2,y2,conf,track_id], ...]}}

track_id = -1 nếu tracker chưa gán (box conf thấp). Dữ liệu này chỉ là "gợi ý"
cho bước 2; mọi box cuối cùng đều được kiểm tra lại bằng mắt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / ".lab-local" / "mot17-09"
OUT = Path(__file__).resolve().parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1920)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    model = YOLO(args.model)
    tracker_model = YOLO(args.model)
    frames = sorted(FRAMES.glob("*.jpg"))[:150]
    result: dict[int, list] = {}

    raw_cache = OUT / "detections_raw.json"
    # 1) Detection thô (không tracker) — giữ cả box conf thấp để bắt người sau kính / bị che.
    for i, f in enumerate(frames if not raw_cache.exists() else []):
        r = model.predict(str(f), conf=args.conf, classes=[0], imgsz=args.imgsz, verbose=False)[0]
        result[i] = [[*map(float, b.xyxy[0].tolist()), float(b.conf[0]), -1] for b in r.boxes]
        if i % 25 == 0:
            print(f"detect frame {i}: {len(result[i])} box", flush=True)

    # 2) Tracker BoT-SORT để có gợi ý liên kết ID giữa các frame.
    if raw_cache.exists():
        result = {int(k): v for k, v in json.loads(raw_cache.read_text()).items()}
    else:
        raw_cache.write_text(json.dumps(result))
    tracked: dict[int, list] = {}
    for i, f in enumerate(frames):
        r = tracker_model.track(str(f), conf=args.conf, classes=[0], imgsz=args.imgsz,
                                tracker="botsort.yaml", persist=True, verbose=False)[0]
        ids = r.boxes.id.tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
        tracked[i] = [[*map(float, b.xyxy[0].tolist()), float(b.conf[0]), int(t)] for b, t in zip(r.boxes, ids)]

    (OUT / "detections.json").write_text(json.dumps({"raw": result, "tracked": tracked}))
    print("saved", OUT / "detections.json")


if __name__ == "__main__":
    main()
