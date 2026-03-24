#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel


def load_segments(caption_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    data = json.loads(caption_path.read_text(encoding="utf-8"))
    segs = data.get("segments", [])
    if not isinstance(segs, list) or len(segs) == 0:
        raise ValueError(f"{caption_path}: no segments")

    texts, starts, ends = [], [], []
    for s in segs:
        a = int(s["start_sec"])
        b = int(s["end_sec"])
        desc = (s.get("description") or s.get("text") or "").strip()
        if not desc or a >= b:
            continue
        texts.append(desc)
        starts.append(a)
        ends.append(b)

    if not texts:
        raise ValueError(f"{caption_path}: all segments empty/invalid")

    return texts, np.array(starts, dtype=np.int32), np.array(ends, dtype=np.int32)


@torch.no_grad()
def embed_texts(model: CLIPModel, processor: CLIPProcessor, texts: List[str], device: str, batch_size: int = 64):
    all_embs = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        all_embs.append(feats.detach().cpu().numpy().astype(np.float32))
        done = min(i + batch_size, n)
        print(f"  [EMB] {done}/{n}", flush=True)
    return np.concatenate(all_embs, axis=0)


def pick_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        print("[WARN] --device cuda requested but cuda not available -> cpu")
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--captions_dir",
        default="data/captions_after_windowing/captions_1",
    )
    ap.add_argument(
        "--out_dir",
        default="data/text_features/clip/text_features_clip_1",
    )
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    captions_dir = Path(args.captions_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_ids = args.only if args.only else sorted([p.stem for p in captions_dir.glob("*.json")])
    if not video_ids:
        raise RuntimeError(f"No caption json files found in {captions_dir}")

    device = pick_device(args.device)

    print(f"[INFO] captions_dir={captions_dir}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] device={device}")
    print(f"[INFO] videos={len(video_ids)} overwrite={args.overwrite}")

    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()

    n_ok = n_skip = n_fail = 0

    for i, vid in enumerate(video_ids, 1):
        cap_path = captions_dir / f"{vid}.json"
        out_path = out_dir / f"{vid}.npz"

        print(f"\n=== ({i}/{len(video_ids)}) {vid} ===")
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] exists: {out_path}")
            n_skip += 1
            continue

        try:
            texts, starts, ends = load_segments(cap_path)
            print(f"[INFO] segments={len(texts)}")

            embs = embed_texts(model, processor, texts, device=device, batch_size=args.batch_size)

            assert embs.shape[0] == len(texts)
            assert embs.dtype == np.float32
            assert starts.shape[0] == ends.shape[0] == embs.shape[0]

            np.savez_compressed(out_path, embeddings=embs, start_sec=starts, end_sec=ends)
            print(f"[OK] saved {out_path}")
            print(f"[OK] embeddings shape={embs.shape} dtype={embs.dtype}")
            n_ok += 1
        except Exception as e:
            print(f"[FAIL] {vid}: {repr(e)}")
            n_fail += 1

    print(f"\n[DONE] ok={n_ok} skip={n_skip} fail={n_fail} total={len(video_ids)}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()