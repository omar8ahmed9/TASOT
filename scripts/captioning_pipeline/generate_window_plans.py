import json
from pathlib import Path
import numpy as np

def make_plan(video_id: str, T: int, window_sec: int = 300):
    windows = []
    for start in range(0, T, window_sec):
        end = min(start + window_sec, T)
        windows.append({
            "start_sec": int(start),
            "end_sec": int(end),
            "name": f"{start:06d}_{end:06d}"
        })
    return {
        "video_id": video_id,
        "duration_sec": int(T),
        "window_sec": int(window_sec),
        "windows": windows
    }

def main():
    feat_dir = Path("data/visual_features/phase")
    out_dir = Path("configs/window_plans")
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_files = sorted(feat_dir.glob("*.npy"))
    if not feat_files:
        raise FileNotFoundError(f"No .npy files found in {feat_dir}")

    window_sec = 300
    count = 0
    for fp in feat_files:
        vid = fp.stem
        x = np.load(fp, mmap_mode="r")
        T = int(x.shape[0])
        plan = make_plan(vid, T, window_sec=window_sec)
        (out_dir / f"{vid}.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        count += 1

    print(f"[OK] Wrote {count} window plans to {out_dir}")
    # show one example
    ex = json.loads((out_dir / f"{feat_files[0].stem}.json").read_text(encoding="utf-8"))
    print("[EX] Example:", ex["video_id"], "windows=", len(ex["windows"]), "last=", ex["windows"][-1]["name"])

if __name__ == "__main__":
    main()
