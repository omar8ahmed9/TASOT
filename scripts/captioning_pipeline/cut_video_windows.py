#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

def cut_clip(src_mp4: Path, dst_mp4: Path, start: int, end: int):
    """Cut [start, end) seconds from src into dst using ffmpeg."""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if dst_mp4.exists():
        return  # resume-friendly

    dur = max(1, int(end) - int(start))
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(int(start)),
        "-t", str(dur),
        "-i", str(src_mp4),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst_mp4),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    plan_dir = Path("configs/window_plans")
    video_root = Path("data/videos")
    out_root = Path("data/windows")

    plan_files = sorted(plan_dir.glob("*.json"))
    if not plan_files:
        raise RuntimeError(f"No plans found in {plan_dir}")

    out_root.mkdir(parents=True, exist_ok=True)

    for i, plan_path in enumerate(plan_files, 1):
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        vid = plan["video_id"]
        windows = plan["windows"]

        src_mp4 = video_root / f"{vid}.mp4"
        if not src_mp4.exists():
            print(f"[MISS] {vid}: missing {src_mp4}")
            continue

        vid_out_dir = out_root / vid
        vid_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== ({i}/{len(plan_files)}) CUT {vid} windows={len(windows)} ===")
        for w in windows:
            name = w["name"]
            start = int(w["start_sec"])
            end = int(w["end_sec"])
            dst_mp4 = vid_out_dir / f"{name}.mp4"

            if dst_mp4.exists():
                continue

            print(f"  [CUT] {vid} {name} ({start}->{end})")
            try:
                cut_clip(src_mp4, dst_mp4, start, end)
            except subprocess.CalledProcessError as e:
                print(f"  [FAIL] {vid} {name}: ffmpeg failed ({e})")

    print("\n[OK] Finished cutting windows for all videos.")

if __name__ == "__main__":
    main()