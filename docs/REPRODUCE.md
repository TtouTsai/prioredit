# Reproduction map

Every table, figure and headline number in the paper, with the command that produces it and an
honest status. Three tiers:

* **A — self-contained.** Runs on this release alone (shipped generations + labels). No GPU.
* **B — needs external resources.** Requires the model weights, the LLM-RG4 checkout, and/or the
  image corpus, all obtainable but not redistributable here.
* **C — script not preserved.** The number is correct and its inputs are shipped, but the
  one-off analysis script that computed it was written in a scratch directory and was not kept.
  The recipe is described so it can be rewritten; we flag these rather than imply full coverage.

```bash
export PRIOREDIT_ROOT=$PWD
P=data/labels/prior_chexpert_labels.json
G=data/labels/gt_chexpert_labels.json
GEN=data/generations
```

---

## Tier A — runs on this release alone

### Table 1, TCA columns + Table 4 (confusion matrices) + §6.2 per-category precision

```bash
python scripts/metric/temporal_accuracy.py --prior $P --gt $G \
  --preds PriorEdit:$GEN/prioredit.json sw:$GEN/llmrg4_sw.json sn:$GEN/llmrg4_sn.json \
          CXRMate:$GEN/cxrmate_withprior.json MLRG:$GEN/mlrg_withprior.json
```

Produces: change P/R (44% / 28% / 30% / 20% / 19%), per-category new/resolved/persistent
precision, resolved true-positives (138 vs 2), and the asserted-vs-true transition counts of
Table 4. Verified to reproduce the paper exactly.

### §6.1 / Table 2, temporal hallucination rate

```bash
python scripts/metric/temporal_halluc_official.py --pred $GEN/prioredit.json \
  --ann $PRIOREDIT_LLMRG4/data/swcmp/annotation.json --prior_labels $P --name PriorEdit
```

The rate itself needs only the prior labels, but the script reads the study list and prior text
from the LLM-RG4 `swcmp` annotation, so in practice this is **Tier B** (see below). The
prior-*withheld* columns of Table 2 use `cxrmate_noprior.json` / `mlrg_noprior.json` the same way.

### App. E, absolute-error accounting and strict `new` analysis (E3 / E4)

```bash
python scripts/analysis/rebuttal_analysis.py
```

Self-contained: reads the shipped labels and all four generation files.

### App. F, qualitative example selection (Figure 2 candidates)

```bash
python scripts/analysis/qualitative_examples/build_fig2_tables.py
```

Needs the `swcmp` annotation for the source report text → Tier B. `editor_paths.py` reconstructs
which editor branch fired per (study, finding) and is validated by regenerating the adopted
`prioredit.json` byte-for-byte.

---

## Tier B — needs external resources

Set `PRIOREDIT_LLMRG4`, `PRIOREDIT_DATA` first (see `docs/DATA.md`).

### PriorEdit end-to-end (paper §3, the pipeline that produced `prioredit.json`)

Three steps. The adopted flags are exactly those recorded below — no others.

```bash
# 1. image-only draft: LLM-RG4 in its no-prior `sn` scenario (LLM-RG4's own inference entry point)
#    -> save/swcmp_sn_off/result/test_result.json   (shipped as $GEN/llmrg4_sn.json)

# 2. confidence probe c_f(I), Eq. 1 — one teacher-forced forward pass per study (GPU)
python scripts/method/probe_confidence.py        # -> finding_confidence.json (shipped)

# 3. confidence-gated rule editor, Eq. 3 + the tag table
python scripts/method/rule_editor.py \
  --drafts $GEN/llmrg4_sn.json \
  --ann    $PRIOREDIT_LLMRG4/data/swcmp/annotation.json \
  --out    out/prioredit.json \
  --confidence data/finding_confidence.json \
  --thresholds configs/confidence_thresholds.json \
  --inplace --high_conf --keep_unchanged --sev_jump 2
```

Step 3 is CPU-only and reproduces the shipped `prioredit.json` byte-for-byte.

### Table 1, presence (CheXbert-14 F1) and RaTE

```bash
python scripts/metric/paper_chexbert_eval.py <eval_dir>   # LLM-RG4's own CheXbert impl
python scripts/metric/radeval_score.py       <eval_dir>   # RaTEScore, Temporal-F1
```

Each `eval_dir` holds `pred.csv` / `gt.csv` with a `Report Impression` column;
`scripts/metric/eval_stage2_radeval.py` builds one from a generation file. Needs the CheXbert
checkpoint (presence) and the RadEval environment (RaTE).

### Table 2, baseline generations (CXRMate, MLRG; prior present and withheld)

```bash
python scripts/baselines/run_cxrmate.py --ann <swcmp annotation> --out out/cxrmate.json
bash   scripts/baselines/run_mlrg.sh   <annotation.json> <version-tag>
```

Both need the released checkpoints and the image corpus. Our MLRG run matches the authors'
released generations' hallucination rate exactly (41.6% vs 41.6%) — the pipeline validation
reported in §5. MLRG's official test protocol is prior-image-only; `run_mlrg.sh` documents why.

### App. D, MS-CXR-T direction probe

```bash
python scripts/direction_probe/prep_mscxrt.py              # build the eval manifest
python scripts/direction_probe/build_direction_trainset.py # silver labels, MS-CXR-T subjects excluded
python scripts/direction_probe/train_direction_verifier.py --epochs 4
python scripts/direction_probe/predict_direction_mscxrt.py --current gold   # 78.4 ceiling
python scripts/direction_probe/predict_direction_mscxrt.py --current $GEN/llmrg4_sn.json  # 36.1
```

Needs MS-CXR-T, the image corpus, CXR-BERT weights, and a GPU.

### App. C, confidence-coverage counts (338 / 54 / precision 61%)

Derived from `data/finding_confidence.json` + `configs/confidence_thresholds.json` +
`data/labels/`, by applying Eq. 3's scope (prior present ∧ draft silent). `rule_editor.py`'s
`confidence_state()` is the reference implementation; `scripts/analysis/qualitative_examples/editor_paths.py`
re-derives the branch taken per (study, finding).

---

## Tier C — inputs shipped, one-off script not preserved

These numbers were computed in a scratch directory that was not version-controlled and is gone.
The inputs needed to recompute them **are** in this release; the recipes are:

| paper number | recipe |
|---|---|
| Presence bootstrap 95% CIs — §6.2, Table 1 note (PriorEdit−`sw` +0.22 CI [−0.79,+1.23]; PriorEdit−`sn` +1.11 CI [+0.69,+1.52]) | study-level bootstrap, B=2000, over CheXbert-14 micro-F1 from `paper_chexbert_eval.py`, n=1698 |
| Paired change-precision CI — §6.2 (+15.5 CI [+7.6,+22.8]) and persistent (+19.2 CI [+16.3,+22.2]) | paired study-level bootstrap, B=2000, resampling the 1337 with-prior studies and rescoring both systems with `temporal_accuracy.score()` on each resample |
| Original-GT split presence — §6.2 (59.23 vs 60.94, −1.71 CI [−2.75,−0.69], n=1576) | same bootstrap against the original MIMIC-CXR GT instead of the RG4-reconstructed GT |
| Two-labeler-agreement subset — §6.2 (92.9% agreement; PriorEdit 44% vs `sw` 28%, 124 vs 1 resolved) | run neural CheXbert over the prior and current GT reports, intersect present-status agreement with the official labeler, restrict `temporal_accuracy.score()` to that subset. `scripts/analysis/e2_independent_labeler.py` does exactly this and is preserved — it needs the neural label files, regenerable with CheXbert; point `PRIOREDIT_NEURAL_LABELS` at them |
| Figure 3, change precision vs volume | x = `#pred` and y = precision of the `change(micro)` row per model from `temporal_accuracy.py` above; error bars are Wilson binomial 95% CIs. All five points are printed by the Tier-A command |
| App. G, LLM-editor ablation — §7 (35.6% hallucination, 13% change precision on 300 studies) | prompt the same LoRA Vicuna backbone as a text rewriter (image-only draft + prior report → add comparison sentences), then score with `temporal_accuracy.py` / `temporal_halluc_official.py` on the same 300 ids |
| App. G ablations: N=5 resampling, ROI crops, region-hint prompting, stacking; severity-jump Δ≥1 vs Δ≥2 | `rule_editor.py` exposes `--samples/--vote_k` (resampling) and `--sev_jump` (severity); the ROI and region-hint variants required generation runs whose driver scripts were not kept |

`temporal_accuracy.py` is importable, so each bootstrap above is a short loop over
`score()`; nothing about the metric itself is missing.
