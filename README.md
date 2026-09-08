# PriorEdit — Prior-Conditioned Report Generators Hallucinate Temporal Change

Code and evaluation artifacts for the ML4H 2026 submission *“Prior-Conditioned Report
Generators Hallucinate Temporal Change: A Direction-Aware Metric and a Decoupled Remedy.”*

Two contributions, both runnable here:

* **TCA (Temporal Change Accuracy)** — a direction-aware, non-circular metric that scores the
  transition a report *asserts* against the *true* transition, obtained by labeling the prior and
  current ground-truth reports independently (paper §4).
* **PriorEdit** — an image-only base drafts the findings, then a confidence-gated rule editor
  consults the prior text only to ground each temporal tag. The prior never enters the
  generator (paper §3).

## Quick start: reproduce the main temporal result

No GPU, no model weights, and no image data are needed — the released generations and labels are
enough to recompute every temporal number in the paper.

```bash
export PRIOREDIT_ROOT=$PWD

python scripts/metric/temporal_accuracy.py \
  --prior data/labels/prior_chexpert_labels.json \
  --gt    data/labels/gt_chexpert_labels.json \
  --preds PriorEdit:data/generations/prioredit.json \
          sw:data/generations/llmrg4_sw.json \
          sn:data/generations/llmrg4_sn.json \
          CXRMate:data/generations/cxrmate_withprior.json \
          MLRG:data/generations/mlrg_withprior.json
```

This prints Table 1's TCA columns and Table 4's confusion counts, on the 1337 studies with a real
prior. Expected (paper Table 1):

| method | change P | change R | new P | resolved P | persistent P | resolved TP |
|---|---:|---:|---:|---:|---:|---:|
| PriorEdit | 44% | 8% | 37% | 49% | 55% | 138 |
| LLM-RG4 `sw` | 28% | 2% | 30% | 12% | 36% | 2 |
| LLM-RG4 `sn` | 30% | 0% | 26% | 100% (1/1) | 33% | 1 |
| CXRMate | 20% | 1% | 19% | 21% | 38% | 3 |
| MLRG | 19% | 0% | 19% | 14% | 41% | 1 |

## Repository layout

```
configs/confidence_thresholds.json   per-finding calibration (paper App. C, Table 6)
data/labels/                         official CheXpert labeler states — the TCA ground truth
  prior_chexpert_labels.json           prior report state, 1337 with-prior studies
  gt_chexpert_labels.json              current GT report state, same studies
  official_labels_by_hash.json         text-hash -> labels cache (labeler memoization)
data/generations/                    the generated reports every table is scored on
  prioredit.json                        PriorEdit (ours)
  llmrg4_sw.json / llmrg4_sn.json       LLM-RG4 with prior text / image-only base
  cxrmate_withprior.json / _noprior     CXRMate, prior present / withheld (Table 2)
  mlrg_withprior.json / _noprior        MLRG, prior present / withheld (Table 2)
data/finding_confidence.json         cached confidence probe c_f(I), Eq. 1
scripts/paths.py                     every filesystem root, from PRIOREDIT_* env vars
scripts/metric/                      TCA + hallucination + presence/RaTE metrics
scripts/method/                      PriorEdit: state extractor, confidence probe, rule editor
scripts/baselines/                   CXRMate and MLRG baseline generation
scripts/direction_probe/             MS-CXR-T direction probe (App. D)
scripts/analysis/                    robustness re-analyses and qualitative example selection
docs/                                environments, data setup, per-claim reproduction map
```

## Paths and environments

All roots come from environment variables, resolved in `scripts/paths.py`:

| variable | meaning | default |
|---|---|---|
| `PRIOREDIT_ROOT` | this checkout | inferred from `scripts/paths.py` |
| `PRIOREDIT_DATA` | corpora root (MIMIC-CXR, MIMIC-RG4, MS-CXR-T, Chest ImaGenome) | `$PRIOREDIT_ROOT/external/data` |
| `PRIOREDIT_LLMRG4` | LLM-RG4 checkout (base model, annotations) | `$PRIOREDIT_ROOT/external/LLM-RG4` |
| `PRIOREDIT_OUT` | regenerable outputs | `$PRIOREDIT_ROOT/out` |

Development used three conda environments; set `PRIOREDIT_PY_MODEL`, `PRIOREDIT_PY_METRIC` and
`PRIOREDIT_PY_LABEL` if yours are not the active interpreter. See `docs/ENVIRONMENTS.md`.

## Data

**No patient data is redistributed here.** MIMIC-CXR requires a PhysioNet credentialed Data Use
Agreement; MIMIC-RG4, MS-CXR-T and Chest ImaGenome come from their own authors. The shipped
`data/labels/` files contain only 14-way CheXpert *label states* keyed by study id — no report
text — and `data/generations/` contains only model-*generated* text. See `docs/DATA.md` for the
expected layout and where each resource comes from.

## Reproducing the rest of the paper

`docs/REPRODUCE.md` maps each table, figure and headline number to the exact command, and states
plainly which numbers are recomputable from this release and which require the model weights,
the image corpus, or an analysis script that was not preserved.

## Licence

Code is released for review under the MIT licence (`LICENSE`). The datasets and the third-party
checkpoints (LLM-RG4, CXRMate, MLRG, CheXbert, CXR-BERT) keep their original licences.
