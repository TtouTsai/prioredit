# Environments

Development used three separate Python environments because the metric stack and the model stack
have conflicting pins. Point the corresponding variable at each interpreter, or run everything in
one environment if your versions allow it.

| variable | environment | used by |
|---|---|---|
| `PRIOREDIT_PY_MODEL` | model inference: Python 3.10, torch with CUDA matching your GPU | `probe_confidence.py`, `run_cxrmate.py`, `run_mlrg.sh`, `paper_chexbert_eval.py` |
| `PRIOREDIT_PY_METRIC` | RadEval: Python 3.11 | `radeval_score.py`, `eval_stage2_radeval.py` |
| `PRIOREDIT_PY_LABEL` | labeling / analysis: Python 3.10 | `chexpert_label.py`, `temporal_assessment.py`, `train_direction_verifier.py`, `scripts/analysis/*` |

All three default to `python`. Pinned package versions for each group are in `requirements.txt`.

## The Tier-A path needs almost nothing

The self-contained reproduction in `docs/REPRODUCE.md` (TCA, the confusion matrices, the
error accounting) uses only the standard library. `scripts/analysis/change_class_analysis.py` and
`temporal_assessment.py` run under a bare Python 3.10+ with no third-party packages.

## Additional requirements by task

* **presence metric** — the CheXbert checkpoint and LLM-RG4's own labeler code, so that the
  CheXbert-14 F1 uses the same implementation as the LLM-RG4 paper.
* **RaTE / Temporal-F1** — `radeval` (2.2.x). RadGraph is skipped: the pinned release has a
  vendored import bug.
* **direction probe** — `torch`, `transformers`, CXR-BERT weights, one GPU.
* **TCA ground truth** — Docker, for the official CheXpert labeler image.
* **note on the base model** — LLM-RG4's SFT checkpoints emit EOS early; its inference code forces
  `min_new_tokens` (50). Keep that when regenerating drafts, or the drafts truncate.
