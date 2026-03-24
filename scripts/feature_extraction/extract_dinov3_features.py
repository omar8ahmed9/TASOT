#!/usr/bin/env python3
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# -------------------- USER PATHS --------------------
# Update VIDEO_DIR to point to the folder containing your input .mp4 videos.
# Example:
#   data/videos/cholec80
# Make sure the folder exists and contains one .mp4 file per video.

VIDEO_DIR = Path("data/videos/cholec80")

# Update OUT_DIR to the folder where extracted DINOv3 features will be saved.
# One .npy file will be written per video.
# Change the dataset subfolder if you are using a dataset other than Cholec80.

OUT_DIR = Path("data/visual_features/dinov3/cholec80")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------

BAD_LOG = OUT_DIR / "bad_videos.txt"


def list_videos():
    return sorted(VIDEO_DIR.glob("*.mp4"))


def get_duration_sec(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot_open")
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps and fps > 0 and nframes and nframes > 0:
        dur = nframes / fps
        return int(math.floor(dur))

    raise RuntimeError("cannot_estimate_duration")


def main():
    videos = list_videos()
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in {VIDEO_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)
    print("[INFO] videos:", len(videos))
    print("[INFO] in:", VIDEO_DIR)
    print("[INFO] out:", OUT_DIR)

    name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    def embed_batch(pil_images):
        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs)
            feats = out.last_hidden_state[:, 0, :]
        return feats.detach().cpu().numpy().astype(np.float32)

    report = {
        "saved": [],
        "skipped_existing": [],
        "failed": {},
        "fps_sampling": 1,
        "model": name,
    }

    for k, vp in enumerate(videos, 1):
        vid = vp.stem
        out_path = OUT_DIR / f"{vid}.npy"

        if out_path.exists():
            report["skipped_existing"].append(vid)
            continue

        try:
            T = get_duration_sec(vp)
        except Exception as e:
            msg = f"duration_error: {repr(e)}"
            print(f"[WARNING] {vid} failed: {msg}")
            report["failed"][vid] = msg
            BAD_LOG.write_text((BAD_LOG.read_text() if BAD_LOG.exists() else "") + f"{vid}\n")
            continue

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            msg = f"cannot_open: {vp}"
            print(f"[WARNING] {vid} failed: {msg}")
            report["failed"][vid] = msg
            BAD_LOG.write_text((BAD_LOG.read_text() if BAD_LOG.exists() else "") + f"{vid}\n")
            continue

        print(f"\n[INFO] ({k}/{len(videos)}) {vid}: T={T} sec (1 fps)")

        feats = np.zeros((T, 768), dtype=np.float32)
        batch_size = 64
        batch_imgs = []
        batch_idx = []

        def flush_batch():
            if not batch_imgs:
                return
            y = embed_batch(batch_imgs)
            for j, idx in enumerate(batch_idx):
                feats[idx] = y[j]
            batch_imgs.clear()
            batch_idx.clear()

        for i in range(T):
            sec = i
            cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
            ok, frame_bgr = cap.read()

            if not ok or frame_bgr is None:
                if i > 0:
                    feats[i] = feats[i - 1]
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            batch_imgs.append(img)
            batch_idx.append(i)

            if len(batch_imgs) >= batch_size:
                flush_batch()

            if (i + 1) % 600 == 0:
                print(f"  [INFO] processed {i+1}/{T} sec")

        flush_batch()
        cap.release()

        np.save(out_path, feats)
        report["saved"].append(vid)
        print(f"[OK] saved {out_path} shape={feats.shape}")

    rep_path = OUT_DIR / "features_extraction_report_dinov3_cholec80.json"
    rep_path.write_text(json.dumps(report, indent=2))
    print("\n[DONE] Report:", rep_path)
    print("[DONE] Saved:", len(report["saved"]),
          "Skipped existing:", len(report["skipped_existing"]),
          "Failed:", len(report["failed"]))


if __name__ == "__main__":
    main()