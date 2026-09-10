# Reproduction map

Every table, figure and headline number in the paper, with the command that produces it and an
honest status. Three tiers:

* **A — self-contained.** Runs on this release alone (shipped generations + labels). No GPU.
* **B — needs external resources.** Requires the model weights, the LLM-RG4 checkout, and/or the
  image corpus, all obtainable but not redistributable here.
* **C — recompute from shipped inputs.** The number is correct and its inputs are shipped; a
  preserved script under `scripts/analysis/` recomputes it (a couple remain short loops over
  `temporal_assessment.score()`, described inline).

```bash
export PRIOREDIT_ROOT=$PWD
P=data/labels/prior_chexpert_labels.json
G=data/labels/gt_chexpert_labels.json
GEN=data/generations
```

---

## Tier A — runs on this release alone

### Table 3, TCA columns + Table 6 (confusion matrices) + §6.2 per-category precision

```bash
python scripts/metric/temporal_assessment.py --prior $P --gt $G \
  --preds PriorEdit:$GEN/prioredit.json sw:$GEN/llmrg4_sw.json sn:$GEN/llmrg4_sn.json \
          CXRMate:$GEN/cxrmate_withprior.json MLRG:$GEN/mlrg_withprior.json
```

Produces: change P/R (44% / 28% / 30% / 20% / 19%), per-category new/resolved/persistent
precision, resolved true-positives (138 vs 2), and the asserted-vs-true transition counts of
Table 6. Verified to reproduce the paper exactly.

### §6.1 / Table 1, temporal hallucination rate

```bash
python scripts/metric/temporal_halluc_official.py --pred $GEN/prioredit.json \
  --ann $PRIOREDIT_LLMRG4/data/swcmp/annotation.json --prior_labels $P --name PriorEdit
```

The rate itself needs only the prior labels, but the script reads the study list and prior text
from the LLM-RG4 `swcmp` annotation, so in practice this is **Tier B** (see below). The
prior-*withheld* columns of Table 1 use `cxrmate_noprior.json` / `mlrg_noprior.json` the same way.

### App. E, absolute-error accounting and strict `new` analysis

```bash
python scripts/analysis/change_class_analysis.py
```

Self-contained: reads the shipped labels and all four generation files.

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

### Table 3, presence (CheXbert-14 F1) and RaTE

```bash
python scripts/metric/paper_chexbert_eval.py <eval_dir>   # LLM-RG4's own CheXbert impl
python scripts/metric/radeval_score.py       <eval_dir>   # RaTEScore, Temporal-F1
```

Each `eval_dir` holds `pred.csv` / `gt.csv` with a `Report Impression` column;
`scripts/metric/eval_stage2_radeval.py` builds one from a generation file. Needs the CheXbert
checkpoint (presence) and the RadEval environment (RaTE).

### Baseline generations (CXRMate, MLRG; prior present and withheld)

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
`confidence_state()` is the reference implementation for the branch taken per (study, finding).

---

## Tier C — inputs shipped; recompute with the analysis scripts

These numbers are recomputed from shipped inputs. Most now have a preserved script under
`scripts/analysis/` (named below); a couple remain short loops over `temporal_assessment.score()`.

| paper number | how to recompute |
|---|---|
| Presence bootstrap 95% CIs — §6.2, Table 3 note (PriorEdit−`sw` +0.22 CI [−0.79,+1.23]; PriorEdit−`sn` +1.11 CI [+0.69,+1.52]) | `scripts/analysis/bootstrap_presence_ci.py --canonical PriorEdit:<dir> sw:<dir> sn:<dir>` (study-level bootstrap, B=2000, CheXbert-14 micro-F1, n=1698) |
| Original-GT split presence — §6.2 (59.23 vs 60.94, −1.71 CI [−2.75,−0.69], n=1576) | same script, `--orig PriorEdit:<dir> sw:<dir>` (against the original MIMIC-CXR GT) |
| Paired change-precision CI — §6.2 (+15.5 CI [+7.6,+22.8]) and persistent (+19.2 CI [+16.3,+22.2]) | paired study-level bootstrap, B=2000, resampling the 1337 with-prior studies and rescoring both systems with `temporal_assessment.score()` on each resample |
| Two-labeler-agreement subset — §6.2 (92.9% agreement; PriorEdit 44% vs `sw` 28%, 124 vs 1 resolved) | `scripts/analysis/neural_chexbert_labels.py` writes the neural prior/GT labels; `scripts/analysis/independent_labeler.py` restricts `temporal_assessment.score()` to the agreement subset (point `PRIOREDIT_NEURAL_LABELS` at them) |
| Independent-parser robustness — App. E (GT self 39–45% across parsers; gap +15.5 regex / +13.8 Qwen3 / +12.8 MedGemma) | `scripts/analysis/llm_parser_eval.py --tag qwen3` (and `--tag medgemma`) re-scores from the shipped text-free caches in `data/llm_parser/` — no model needed. Regenerate with `llm_parser_extract.py` (local Ollama; report text stays on-machine per the DUA) |
| Figure 2, change precision vs volume | x = `#pred` and y = precision of the `change(micro)` row per model from `temporal_assessment.py` above; error bars are Wilson binomial 95% CIs. All five points are printed by the Tier-A command |
| App. F, LLM-editor ablation — §7 (35.6% hallucination, 13% change precision on 300 studies) | `scripts/analysis/llm_editor_ablation.py` generates with the LoRA Vicuna backbone; score with `temporal_assessment.py` / `temporal_halluc_official.py` on the produced 300 ids |
| App. C / F, threshold operating point (P* sweep) | `scripts/analysis/risk_coverage_sweep.py` — change precision stays flat across P* while the `unchanged` volume grows |
| App. F ablations: N=5 resampling, ROI crops, region-hint prompting, stacking; severity-jump Δ≥1 vs Δ≥2 | `rule_editor.py` exposes `--samples/--vote_k` (resampling) and `--sev_jump` (severity); the ROI and region-hint variants required generation runs whose driver scripts were not kept |

`temporal_assessment.py` is importable, so each bootstrap above is a short loop over
`score()`; nothing about the metric itself is missing.
