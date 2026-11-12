# PriorEdit — Prior-Conditioned Report Generators Hallucinate Temporal Change

Code and evaluation artifacts for *"Prior-Conditioned Report Generators Hallucinate Temporal
Change: A Direction-Aware Metric and a Decoupled Remedy,"* published in the Proceedings of
Machine Learning for Health (ML4H) 2026.

Authors: Yan-Hong Chen, Yun-Chieh Tsai, Po-Chih Kuo


Two contributions, both runnable here:

* **TCA (Temporal Change Assessment)** — a direction-aware, non-circular metric that scores the
  transition a report *asserts* against the *true* transition, obtained by labeling the prior and
  current ground-truth reports independently (paper §3).
* **PriorEdit** — an image-only base drafts the findings, then a confidence-gated rule editor
  consults the prior text only to ground each temporal tag. The prior never enters the
  generator (paper §6).

## Quick start: reproduce the main temporal result

No GPU, no model weights, and no image data are needed — the released generations and labels are
enough to recompute the main temporal results: the TCA columns of Table 2, Tables 1 (temporal
hallucination), 4, 7, 9 and 10, the Change column of Table 8, the strict change classes, and the
LLM-parser robustness gaps (see `docs/REPRODUCE.md`).

```bash
export PRIOREDIT_ROOT=$PWD

python scripts/metric/temporal_assessment.py \
  --prior data/labels/prior_chexpert_labels.json \
  --gt    data/labels/gt_chexpert_labels.json \
  --preds PriorEdit:data/generations/prioredit.json \
          sw:data/generations/llmrg4_sw.json \
          sn:data/generations/llmrg4_sn.json \
          CXRMate:data/generations/cxrmate_withprior.json \
          MLRG:data/generations/mlrg_withprior.json
```

This prints Table 2's TCA columns, on the 1337 studies with a real prior (the full confusion
matrices of Table 7 come from `scripts/analysis/error_directions.py`). Expected (paper Table 2):

| method | change P | change R | new P | resolved P | persistent P | resolved TP |
|---|---:|---:|---:|---:|---:|---:|
| PriorEdit | 44% | 8% | 37% | 49% | 55% | 138 |
| LLM-RG4 `sw` | 28% | 2% | 30% | 12% | 36% | 2 |
| LLM-RG4 `sn` | 30% | 0% | 26% | 100% (1/1) | 33% | 1 |
| CXRMate | 20% | 1% | 19% | 21% | 38% | 3 |
| MLRG | 19% | 0% | 19% | 14% | 41% | 1 |

## Repository layout

```
prioredit/
├── configs/
│   └── confidence_thresholds.json            per-finding calibration (App. C, Table 6)
│
├── data/
│   ├── labels/                               official CheXpert states — the TCA ground truth
│   │   ├── prior_chexpert_labels.json        prior report, 1337 with-prior studies
│   │   ├── gt_chexpert_labels.json           current GT report, same studies
│   │   └── official_labels_by_hash.json      text-hash → labels cache (memoization)
│   ├── generations/                          the reports every table is scored on
│   │   ├── prioredit.json                    PriorEdit (ours)
│   │   ├── llmrg4_sw.json                    LLM-RG4 with prior text
│   │   ├── llmrg4_sn.json                    LLM-RG4 image-only base
│   │   ├── cxrmate_{withprior,noprior}.json  CXRMate, prior / withheld (Table 1)
│   │   ├── mlrg_{withprior,noprior}.json     MLRG, prior / withheld (Table 1)
│   │   └── qwen3vl_imageonly.json            Qwen3-VL-2B image-only base (App. F, Table 8)
│   ├── llm_parser/                           LLM-parser asserted transitions (App. E)
│   │   ├── asserted_*_qwen3.json             Qwen3-30B-A3B re-extraction
│   │   ├── asserted_*_medgemma.json          MedGemma-27B re-extraction
│   │   └── meta_*.json                       model digest, temperature, parse-fail counts
│   └── finding_confidence.json               cached confidence probe c_f(I), Eq. 1
│
├── scripts/
│   ├── paths.py                              every filesystem root, from PRIOREDIT_* env vars
│   ├── metric/                               TCA + hallucination + presence/RaTE metrics
│   ├── method/                               state extractor, probe, calibration, rule editor
│   ├── baselines/                            CXRMate, MLRG and Qwen3-VL-2B generation
│   ├── direction_probe/                      MS-CXR-T direction probe (App. D)
│   └── analysis/                             bootstrap CIs, robustness, error directions, ablations
│
├── docs/                                     environments, data setup, per-claim reproduction map
├── requirements.txt                          pinned versions (the quick start above needs none)
└── LICENSE
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
`PRIOREDIT_PY_LABEL` if yours are not the active interpreter. Package versions are in
`requirements.txt` (the Tier-A quick start above needs none); see `docs/ENVIRONMENTS.md` for the
three-environment split.

## Data

**No patient data is redistributed here.** MIMIC-CXR requires a PhysioNet credentialed Data Use
Agreement; MIMIC-RG4, MS-CXR-T and Chest ImaGenome come from their own authors. The shipped
`data/labels/` files contain only CheXpert *label states* (13 findings, the 14 CheXpert classes
without No Finding) keyed by the image id of the LLM-RG4 annotation — no report text — and `data/generations/` contains only model-*generated* text. See `docs/DATA.md` for the
expected layout and where each resource comes from.

## Reproducing the rest of the paper

`docs/REPRODUCE.md` maps each table, figure and headline number to the command that produces it.
Where a number needs an input we cannot redistribute (report text, model weights, the image
corpus), the input is named and `docs/DATA.md` says where to obtain it. The few numbers without a
dedicated script are marked there, with the computation spelled out.

## Licence

Code is released under the MIT licence (`LICENSE`). The datasets and the third-party
checkpoints (LLM-RG4, CXRMate, MLRG, Qwen3-VL-2B-RRG-SFT, CheXbert, CXR-BERT) keep their original
licences.
