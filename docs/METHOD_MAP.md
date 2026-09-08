# Paper → code map

Where each formal object in the paper lives in this repository.

## PriorEdit (§3)

| paper | code |
|---|---|
| state extractor `ext(·)_f ∈ {pos, neg, none}` (§3.1) | `scripts/method/state_extractor.py`, `extract()`. Mention lexicon for the five MS-CXR-T findings aligned to the CheXpert labeler's phrase lists; the rest hand-written (App. B) |
| official-labeler variant of `ext` | `state_extractor.py`, `official_extract()` / `load_official()`, backed by `data/labels/official_labels_by_hash.json` |
| severity ordinal (trace/small/mild < moderate < large/severe) | `state_extractor.py`, `SEV` / `sev_in()` — used only for the severity-jump tag |
| image-only base (LLM-RG4 `sn`) | external; its output is shipped as `data/generations/llmrg4_sn.json` |
| confidence `c_f(I) = log P(T⁺\|I) − log P(T⁻\|I)` (Eq. 1) | `scripts/method/probe_confidence.py`; cached in `data/finding_confidence.json` |
| calibration `σ(a_f c + b_f)`, thresholds `τ_pos`, `τ_neg` (Eq. 2) | fitted offline; adopted values in `configs/confidence_thresholds.json` (`tau_pos`, `tau_neg`, `open_pos`, `open_neg`, `n`, `base_in_scope`) — matches Table 6 |
| adopted state `s_f` (Eq. 3) | `scripts/method/rule_editor.py`, `confidence_state()` |
| rule editor: `(ext(R)_f, s_f)` → tag, merged in place | `rule_editor.py`, `edit_inplace()`. `τ_neg` branch emits `resolved` when `s_f = neg` |
| adopted editor configuration | `--confidence --thresholds --inplace --high_conf --keep_unchanged --sev_jump 2` |

Findings left closed for coverage (pleural effusion, consolidation) appear in
`configs/confidence_thresholds.json` with `open_pos: false` — too few clean validation pairs
(App. C).

## TCA (§4)

| paper | code |
|---|---|
| true transition from (prior state, current-GT state) | `scripts/metric/temporal_accuracy.py`, `true_transitions()`; states from `data/labels/{prior,gt}_chexpert_labels.json` |
| asserted transition, polarity-aware parse | `temporal_accuracy.py`, `asserted_transitions()`; `NEW` / `RESOLVED` / `PERSIST` patterns with the `NO_NEW` / `NOT_RESOLVED` suppressions |
| exclusion of static size adjectives (“enlarged”) | same module; documented in App. E — this is what puts the `sn` draft at 0.04 rather than an apparent 0.36 grounded temporal statements per report |
| micro-averaged change precision/recall over {new, resolved} | `temporal_accuracy.py`, `score()`, row `change(micro)` |
| grounded temporal statements per report | the `#pred` totals from `asserted_transitions()` |
| temporal hallucination (claims the prior contradicts) | `scripts/metric/temporal_halluc_official.py` (official labeler as prior truth). `temporal_halluc_eval.py` is the regex-prior variant it builds on; `temporal_halluc_indep.py` uses neural CheXbert instead |

Note the two metrics are deliberately not complementary: hallucination checks a claim only
against the prior, TCA against the true transition (prior × current GT).

## Presence and report quality (§5)

| paper | code |
|---|---|
| CheXbert-14 F1, paper implementation | `scripts/metric/paper_chexbert_eval.py` — reuses LLM-RG4's own labeler and binarization so the numbers are on the LLM-RG4 paper's scale |
| RaTE, Temporal-F1 | `scripts/metric/radeval_score.py` (RadEval) |
| generation file → `pred.csv`/`gt.csv` eval dir | `scripts/metric/eval_stage2_radeval.py` |

## Analyses and appendices

| paper | code |
|---|---|
| App. D, MS-CXR-T direction probe (78.4 ceiling → 36.1 deployment) | `scripts/direction_probe/` |
| App. E, absolute-error accounting + strict `new` | `scripts/analysis/rebuttal_analysis.py` |
| §6.2, independent-labeler robustness | `scripts/analysis/e2_independent_labeler.py` |
| Figure 2 example selection, editor-branch attribution | `scripts/analysis/qualitative_examples/` |

`qualitative_examples/sw_artifact.py` exists for a specific honesty reason: the `new` pattern
contains `appear\w*` and is tested before the persistent branch, so a sentence reading
“contours appear unchanged” parses as `new`. Those rows are excluded from the figure rather than
shown as baseline errors. The metric itself is left unmodified.
