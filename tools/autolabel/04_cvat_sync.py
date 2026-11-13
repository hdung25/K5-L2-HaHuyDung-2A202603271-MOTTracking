"""Bước 4 — Đẩy 5 track lên job CVAT local, xuất MOT 1.1 + CVAT for video 1.1, chuyển job sang completed.

Chạy từ thư mục gốc repo (sẽ hỏi tài khoản CVAT local, mật khẩu nhập ẩn):
    .\\.venv\\Scripts\\python.exe tools/autolabel/04_cvat_sync.py --mssv 2A202603271 --name HaHuyDung
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from getpass import getpass
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import setup_local_lab as lab  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mssv", required=True)
    ap.add_argument("--name", required=True, help="HoVaTen không dấu, dùng đặt tên file xuất")
    ap.add_argument("--server", default="http://localhost:8080")
    a = ap.parse_args()

    from cvat_sdk import make_client

    variant = lab.variant_for(a.mssv)
    task_name = f"DAY05-MOT17-09-{variant}-{lab.student_id(a.mssv)}"
    frames = lab.verify_frames(lab.LOCAL_DATA / "mot17-09")
    tracks_xml = (HERE / "out" / "tracks.xml").read_text(encoding="utf-8")
    exports = ROOT / "src" / "exports"

    user = input("Tên đăng nhập CVAT local: ").strip()
    pwd = getpass("Mật khẩu CVAT local: ")
    with make_client(host=a.server, credentials=(user, pwd)) as client, TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task_id, job_id = lab.open_or_create_job(client, task_name, frames)
        job = client.jobs.retrieve(job_id)
        print(f"Task #{task_id}  Job #{job_id}")

        # 1) lấy khung annotations.xml đúng meta của job (xuất bản rỗng/hiện tại)
        base = tmp / "base.zip"
        job.export_dataset("CVAT for video 1.1", base, include_images=False)
        with zipfile.ZipFile(base) as z:
            xml = z.read("annotations.xml").decode("utf-8")
        head = xml[: xml.index("</meta>") + len("</meta>")]
        new_xml = head + "\n" + tracks_xml + "\n</annotations>\n"
        up = tmp / "upload.zip"
        with zipfile.ZipFile(up, "w") as z:
            z.writestr("annotations.xml", new_xml)

        # 2) import (thay toàn bộ annotation của job)
        job.import_annotations("CVAT 1.1", up)
        n = len(job.get_annotations().tracks)
        print(f"Đã import {n} track vào job #{job_id}")

        # 3) xuất 2 file nộp, không kèm ảnh
        exports.mkdir(parents=True, exist_ok=True)
        mot = exports / f"{a.name}_mot.zip"
        vid = exports / f"{a.name}_cvat_video.zip"
        for old in (mot, vid):  # SDK không ghi đè file có sẵn
            old.unlink(missing_ok=True)
        job.export_dataset("MOT 1.1", mot, include_images=False)
        job.export_dataset("CVAT for video 1.1", vid, include_images=False)
        print("Đã xuất:", mot, vid, sep="\n  ")

        # 4) job -> completed
        job.update({"stage": "acceptance", "state": "completed"})
        job.fetch()
        print(f"Job state: {job.state} | stage: {job.stage}")
        print(f"Job URL: {a.server}/tasks/{task_id}/jobs/{job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
