#!/bin/bash
# Reproduce the MLRG baseline generations (Table 1 / Table 2 rows "MLRG").
#
# This is the authors' own inference entry point (main_v0926_ablation_study.py) from the released
# MLRG checkout, driven with the released MK-runner/MLRG checkpoint. We follow MLRG's OFFICIAL test
# protocol unchanged: --is_prior_report yes is passed, but the released val/test annotation carries
# no prior findings text, so the prior report resolves to MLRG's absence token [NHPR] and the model
# is effectively prior-IMAGE-only at test time (see paper App. G).
#
# Usage:  run_mlrg.sh <annotation.json> <version-tag>
#   with-prior run : the released annotation
#   no-prior  run : the same annotation with the prior scan removed (absence encoding activates)
#
# Required environment (see scripts/paths.py / docs/ENVIRONMENTS.md):
#   MLRG_DIR         the MLRG checkout containing main_v0926_ablation_study.py
#   MLRG_SNAPSHOT    the MK-runner/MLRG HF snapshot dir (checkpoint + view-position embedding)
#   DISTILGPT2_PATH  local distilgpt2 weights
#   PRIOREDIT_DATA   corpora root (MIMIC-CXR-JPG under $PRIOREDIT_DATA/mimic-cxr-jpg/files)
#   PRIOREDIT_LLMRG4 LLM-RG4 checkout (reused here only for its cached hf/ encoders)
#   PRIOREDIT_OUT    output root for the generated reports
set -euo pipefail

ANN="$1"; VERSION="$2"
cd "$MLRG_DIR"

RG="${PRIOREDIT_LLMRG4:?set PRIOREDIT_LLMRG4}"
DATA="${PRIOREDIT_DATA:?set PRIOREDIT_DATA}"
SNAP="${MLRG_SNAPSHOT:?set MLRG_SNAPSHOT}"
OUT="${PRIOREDIT_OUT:-./out}/mlrg"

HF_HOME="${HF_HOME:-$DATA/hf-cache}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PRIOREDIT_PY_MODEL:-python}" main_v0926_ablation_study.py \
  --data_name mimic_cxr --task test --version "$VERSION" \
  --ann_path "$ANN" \
  --view_position_embed "$SNAP/radiology report/five_work_mimic_cxr_view_position_v1.1.json" \
  --images_dir "$DATA/mimic-cxr-jpg/files/" --ckpt_zoo_dir "" \
  --chexbert_path "$RG/hf/chexbert.pth" --bert_path "$RG/hf/bert-base-uncased" \
  --rad_dino_path "$RG/hf/rad-dino" --cxr_bert_path "$RG/hf/BiomedVLP-CXR-BERT-specialized" \
  --distilgpt2_path "${DISTILGPT2_PATH:?set DISTILGPT2_PATH}" --radgraph_path radgraph \
  --test_ckpt_path "$SNAP/mimic-cxr/finetune/best_model.ckpt" \
  --is_multiview_learning yes --is_prior_scan yes --is_prior_report yes --is_indication yes \
  --is_save_checkpoint no --batch_size 6 --num_workers 8 --num_beams 3 \
  --exp_dir_trial "$OUT"
