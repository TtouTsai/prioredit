# Paper → code map

Where each formal object in the paper lives in this repository.

## PriorEdit (§6)

| paper | code |
|---|---|
| state extractor `ext(·)_f ∈ {pos, neg, none}` (§6.1) | `scripts/method/state_extractor.py`, `extract()`. Mention lexicon for the five MS-CXR-T findings aligned to the CheXpert labeler's phrase lists; the rest hand-written (App. B) |
| official-labeler variant of `ext` | `state_extractor.py`, `official_extract()` / `load_official()`, backed by `data/labels/official_labels_by_hash.json` |
| severity ordinal (trace/small/mild < moderate < large/severe) | `state_extractor.py`, `SEV` / `sev_in()` — used only for the severity-jump tag |
| image-only base (LLM-RG4 `sn`) | external; its output is shipped as `data/generations/llmrg4_sn.json` |
| confidence `c_f(I) = log P(T⁺\|I) − log P(T⁻\|I)` (Eq. 1) | `scripts/method/probe_confidence.py`; cached in `data/finding_confidence.json` |
| calibration `σ(a_f c + b_f)`, thresholds `τ_pos`, `τ_neg` (Eq. 3, App. C) | `scripts/method/calibrate.py` fits the logistic on the validation coverage scope, writes `configs/confidence_thresholds.json` (`tau_pos`, `tau_neg`, `open_pos`, `open_neg`, `n`, `base_in_scope`), and prints the Table 6 view (`a_f`, `b_f`, `τ_pos`, `τ_neg`) |
| adopted state `s_f` (Eq. 2) | `scripts/method/rule_editor.py`, `confidence_state()` |
| rule editor: `(ext(R)_f, s_f)` → tag, merged in place | `rule_editor.py`, `edit_inplace()`. `τ_neg` branch emits `resolved` when `s_f = neg` |
| adopted editor configuration | `--confidence --thresholds --inplace --high_conf --keep_unchanged --sev_jump 2` |

Pleural effusion and consolidation are absent from `configs/confidence_thresholds.json`:
`calibrate.py` omits findings with too few clean in-scope validation pairs (App. C). Lung lesion
and atelectasis are fitted but left closed (`open_pos: false`, `open_neg: false`).

## TCA (§3)

| paper | code |
|---|---|
| true transition from (prior state, current-GT state) | `scripts/metric/temporal_assessment.py`, `true_transitions()`; states from `data/labels/{prior,gt}_chexpert_labels.json` |
| asserted transition, polarity-aware parse | `temporal_assessment.py`, `asserted_transitions()`; `NEW` / `RESOLVED` / `PERSIST` patterns with the `NO_NEW` / `NOT_RESOLVED` suppressions |
| exclusion of static size adjectives (“enlarged”) | same module; documented in App. E — this is what puts the `sn` draft at 0.04 rather than an apparent 0.36 grounded temporal statements per report |
| micro-averaged change precision/recall over {new, resolved} | `temporal_assessment.py`, `score()`, row `change(micro)` |
| grounded temporal statements per report | the `#pred` totals from `asserted_transitions()` |
| temporal hallucination (claims the prior contradicts) | `scripts/metric/temporal_halluc_official.py` (official labeler as prior truth). `temporal_halluc_eval.py` is the regex-prior variant it builds on; `temporal_halluc_indep.py` uses neural CheXbert instead |

Note the two metrics are deliberately not complementary: hallucination checks a claim only
against the prior, TCA against the true transition (prior × current GT).

## Presence and report quality (§4)

| paper | code |
|---|---|
| CheXbert-14 F1, paper implementation | `scripts/metric/paper_chexbert_eval.py` — reuses LLM-RG4's own labeler and binarization so the numbers are on the LLM-RG4 paper's scale |
| RaTE, Temporal-F1 | `scripts/metric/radeval_score.py` (RadEval) |
| generation file → `pred.csv`/`gt.csv` eval dir | `scripts/metric/make_eval_dir.py` (from a generation file) or `scripts/metric/eval_stage2_radeval.py` (from an LLM-RG4 inference output dir) |

## Analyses and appendices

| paper | code |
|---|---|
| App. C, Table 6 confidence calibration | `scripts/method/calibrate.py` |
| App. F, threshold operating point (risk-coverage sweep of P*) | `scripts/analysis/risk_coverage_sweep.py` |
| App. D, MS-CXR-T direction probe (78.4 ceiling → 36.1 deployment) | `scripts/direction_probe/` |
| §7 absolute-error accounting (0.75 vs 0.61 errors per report) + strict `new` and `resolved` (§7, App. E) | `scripts/analysis/change_class_analysis.py` |
| App. F, LLM-editor ablation (35.6% halluc / 13% change P) | `scripts/analysis/llm_editor_ablation.py` |
| App. F, same editor on other bases: `sw` (39%, 115 resolved, 21.5% halluc), Qwen3-VL-2B (21%) | `scripts/method/rule_editor.py --inplace --high_conf --keep_unchanged` on `llmrg4_sw.json` / `qwen3vl_imageonly.json` |
| App. F, Table 8 image-only bases (Qwen3-VL-2B vs `sn`) | `scripts/baselines/run_qwen3vl.py` (generation), `make_eval_dir.py --with_prior` + `paper_chexbert_eval.py` (presence), `scripts/analysis/imageonly_bases_rate.py` (RaTE), `temporal_assessment.py` (Change) |
| Table 4, Table 7 (full confusion matrices) and App. H (Tables 9, 10), error directions | `scripts/analysis/error_directions.py` |
| §7 and App. E, independent-labeler robustness | `scripts/analysis/independent_labeler.py`, fed by `scripts/analysis/neural_chexbert_labels.py` (neural prior/GT labels) |
| App. E, independent-parser robustness | `scripts/analysis/llm_parser_extract.py` (local Ollama LLM parser: Qwen3-30B-A3B, MedGemma-27B) + `scripts/analysis/llm_parser_eval.py`; text-free asserted-transition caches shipped in `data/llm_parser/` |
| §7 / Table 2 and App. C, presence bootstrap 95% CIs (main and standard split) | `scripts/analysis/bootstrap_presence_ci.py` |
