#!/usr/bin/env bash
set -e

if [ ! -f "src/train.py" ]; then
  echo "[ERROR] Run this script from the TASOT repo root."
  exit 1
fi

PY=src/train.py
BASE=data/MultiBypass140
GPU=0
EPOCHS=200
BS=1

# ---- feature name (for logging) ----
FEAT_NAME=DINOv3vis_CLIPtxt_MB140

# ---- feature paths ----
VIS_DIR=data/visual_features/dinov3
TXT_DIR=data/text_features/clip/text_features_clip_1

# ---- multimodal hyperparams ----
BETA=0.8
PHASE_K=12
STEP_K=46

CAPTION=clip1

for FOLD in 0 1 2 3 4; do
  echo "=============================="
  echo "Running FOLD $FOLD"
  echo "=============================="

  # ---------- Bern Phase ----------
  python "$PY" \
    -p "$BASE/Bern" \
    -d MB140_Bern_Phase_1fps -ac all \
    --visual-dir "$VIS_DIR" \
    --text-dir "$TXT_DIR" \
    --caption-name "$CAPTION" \
    --use-mm-cost \
    --beta-mm "$BETA" \
    --feature-name "$FEAT_NAME" \
    --fold "$FOLD" \
    --layers 768 512 128 40 \
    --layers-txt 512 512 128 40 \
    --n-clusters "$PHASE_K" \
    --n-epochs "$EPOCHS" \
    --batch-size "$BS" \
    --gpu "$GPU" \
    --std-feats

  # ---------- Bern Step ----------
  python "$PY" \
    -p "$BASE/Bern" \
    -d MB140_Bern_Step_1fps -ac all \
    --visual-dir "$VIS_DIR" \
    --text-dir "$TXT_DIR" \
    --caption-name "$CAPTION" \
    --use-mm-cost \
    --beta-mm "$BETA" \
    --feature-name "$FEAT_NAME" \
    --fold "$FOLD" \
    --layers 768 512 128 40 \
    --layers-txt 512 512 128 40 \
    --n-clusters "$STEP_K" \
    --n-epochs "$EPOCHS" \
    --batch-size "$BS" \
    --gpu "$GPU" \
    --std-feats

  # ---------- Stras Phase ----------
  python "$PY" \
    -p "$BASE/Stras" \
    -d MB140_Stras_Phase_1fps -ac all \
    --visual-dir "$VIS_DIR" \
    --text-dir "$TXT_DIR" \
    --caption-name "$CAPTION" \
    --use-mm-cost \
    --beta-mm "$BETA" \
    --feature-name "$FEAT_NAME" \
    --fold "$FOLD" \
    --layers 768 512 128 40 \
    --layers-txt 512 512 128 40 \
    --n-clusters "$PHASE_K" \
    --n-epochs "$EPOCHS" \
    --batch-size "$BS" \
    --gpu "$GPU" \
    --std-feats

  # ---------- Stras Step ----------
  python "$PY" \
    -p "$BASE/Stras" \
    -d MB140_Stras_Step_1fps -ac all \
    --visual-dir "$VIS_DIR" \
    --text-dir "$TXT_DIR" \
    --caption-name "$CAPTION" \
    --use-mm-cost \
    --beta-mm "$BETA" \
    --feature-name "$FEAT_NAME" \
    --fold "$FOLD" \
    --layers 768 512 128 40 \
    --layers-txt 512 512 128 40 \
    --n-clusters "$STEP_K" \
    --n-epochs "$EPOCHS" \
    --batch-size "$BS" \
    --gpu "$GPU" \
    --std-feats
done