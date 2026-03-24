#!/usr/bin/env python3
import json
import re
import argparse
from pathlib import Path

WIN_RE = re.compile(r"^(\d{6})_(\d{6})\.json$")

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def merge_one_video(video_id: str, in_root: Path, out_root: Path, overwrite: bool = False) -> tuple[bool, str]:
    """
    Merge window caption JSONs for one video_id.
    Returns (ok, message).
    """
    in_dir = in_root / video_id
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{video_id}.json"

    if out_path.exists() and not overwrite:
        return True, f"[SKIP] {video_id}: already merged -> {out_path}"

    if not in_dir.exists():
        return False, f"[MISS] {video_id}: missing dir {in_dir}"

    files = sorted([p for p in in_dir.glob("*.json") if WIN_RE.match(p.name)])
    if not files:
        return False, f"[MISS] {video_id}: no window json files in {in_dir}"

    merged = []
    n_ok_files = 0
    n_bad_files = 0

    for fp in files:
        m = WIN_RE.match(fp.name)
        w_start = int(m.group(1))
        w_end = int(m.group(2))

        try:
            data = load_json(fp)
        except Exception:
            n_bad_files += 1
            continue

        segs = data.get("segments", [])
        if not isinstance(segs, list) or len(segs) == 0:
            continue

        any_valid = False
        for s in segs:
            if "start_sec" not in s or "end_sec" not in s:
                continue
            try:
                a = int(s["start_sec"]) + w_start
                b = int(s["end_sec"]) + w_start
            except Exception:
                continue

            a = max(a, w_start)
            b = min(b, w_end)
            if a >= b:
                continue

            desc = (s.get("description") or s.get("text") or "").strip()
            if not desc:
                continue

            merged.append({
                "start_sec": a,
                "end_sec": b,
                "description": desc,
            })
            any_valid = True

        if any_valid:
            n_ok_files += 1

    if not merged:
        return False, f"[FAIL] {video_id}: produced 0 segments (ok_files={n_ok_files}, bad_files={n_bad_files})"

    merged.sort(key=lambda x: (x["start_sec"], x["end_sec"]))

    cleaned = []
    last = None
    for s in merged:
        key = (s["start_sec"], s["end_sec"], s["description"])
        if key == last:
            continue
        cleaned.append(s)
        last = key

    final = {"video_id": video_id, "fps": 1, "segments": cleaned}
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    return True, f"[OK] {video_id}: {len(cleaned)} segments -> {out_path} (from {len(files)} window files; ok_files={n_ok_files}; bad_files={n_bad_files})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_root",
        default="data/windows_captions_gemini/window_captions_1",
        help="root folder containing per-video window json dirs",
    )
    ap.add_argument(
        "--out_root",
        default="data/captions_after_windowing/captions_1",
        help="output folder for merged captions",
    )
    ap.add_argument("--overwrite", action="store_true", help="re-merge even if output already exists")
    ap.add_argument("--only", nargs="*", default=None, help="optional list of video_ids to merge")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    if not in_root.exists():
        raise FileNotFoundError(f"Missing in_root: {in_root}")

    if args.only:
        video_ids = args.only
    else:
        video_ids = sorted([p.name for p in in_root.iterdir() if p.is_dir()])

    n_ok = n_fail = 0
    for vid in video_ids:
        ok, msg = merge_one_video(vid, in_root, out_root, overwrite=args.overwrite)
        print(msg)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n[DONE] ok={n_ok} fail={n_fail} total={len(video_ids)}")

if __name__ == "__main__":
    main()