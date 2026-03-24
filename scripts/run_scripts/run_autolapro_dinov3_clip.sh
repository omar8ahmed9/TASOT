#!/usr/bin/env bash
set -e

if [ ! -f "src/train.py" ]; then
  echo "[ERROR] Run this script from the TASOT repo root."
  exit 1
fi

PY=src/train.py
GPU=0
EPOCHS=200
BS=1

# ---- dataset base path ----
# This should be the parent directory that contains AutoLaparo_Phase_1fps/
BASE=data

# ---- dataset name ----
DATASET=AutoLaparo_Phase_1fps

# ---- feature name (for logging) ----
FEAT_NAME=DINOv3vis_CLIPtxt_AutoLaparo

# ---- feature paths ----
VIS_DIR=data/visual_features/dinov3/AutoLaparo
TXT_DIR=data/text_features/clip/AutoLaparo/text_features_clip_1

# ---- multimodal hyperparams ----
BETA=0.8
PHASE_K=7

# ---- split root ----
SPLIT_ROOT=splits

# ---- fold ----
FOLD=0

python "$PY" \
  -p "$BASE" \
  -d "$DATASET" -ac all \
  --visual-dir "$VIS_DIR" \
  --text-dir "$TXT_DIR" \
  --caption-name "clip1" \
  --use-mm-cost \
  --beta-mm "$BETA" \
  --feature-name "$FEAT_NAME" \
  --split-root "$SPLIT_ROOT" \
  --fold "$FOLD" \
  --layers 768 512 128 40 \
  --layers-txt 512 512 128 40 \
  --n-clusters "$PHASE_K" \
  --n-epochs "$EPOCHS" \
  --batch-size "$BS" \
  --gpu "$GPU" \
  --std-feats