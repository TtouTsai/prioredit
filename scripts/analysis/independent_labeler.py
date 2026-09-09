#!/usr/bin/env python
"""Independent-labeler robustness (Appendix E): does the PriorEdit vs sw change-precision gap survive an
INDEPENDENT prior labeler?

TCA's prior state uses the official CheXpert RULE labeler, and PriorEdit's editor ext(R) is a regex
aligned to that same CheXpert lexicon (93.8% agreement), a shared oracle on the prior side. This
recomputes TCA's true transition with the prior (and/or current-GT) state taken from the NEURAL CheXbert
labeler (smit2020) instead -- a different labeler the editor is not aligned to -- and re-scores every
model with the unchanged temporal_assessment.score(). The gap holds and in fact widens, so it is not a
shared-oracle artifact.

Neural labels are produced by scripts/analysis/neural_chexbert_labels.py; point PRIOREDIT_NEURAL_LABELS
at them. Env: bare Python 3.10+ once the labels exist; no GPU.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import json, os, sys
import temporal_assessment as TA  # unchanged shared module

RG = paths.LLMRG4
# Neural-CheXbert labels are NOT shipped (regenerate: see docs/REPRODUCE.md, step "neural labels").
TMP = os.environ.get("PRIOREDIT_NEURAL_LABELS", os.path.join(paths.OUT, "neural_labels"))

OFF_PRIOR = json.load(open(os.path.join(paths.LABELS, "prior_chexpert_labels.json")))  # 1337 with-prior
OFF_GT = json.load(open(os.path.join(paths.LABELS, "gt_chexpert_labels.json")))
CXB_PRIOR = json.load(open(f"{TMP}/chexbert_prior_labels.json"))  # neural, 1698
CXB_GT = json.load(open(f"{TMP}/chexbert_gt_labels.json"))
KEYS = set(OFF_PRIOR) & set(OFF_GT)  # the 1337 with-prior studies -> keep all configs on the same set
def restrict(d): return {k: d[k] for k in d if k in KEYS}
CXB_PRIOR, CXB_GT = restrict(CXB_PRIOR), restrict(CXB_GT)

PREDS = {
    "sw": os.path.join(paths.GENERATIONS, "llmrg4_sw.json"),
    "PriorEdit": os.path.join(paths.GENERATIONS, "prioredit.json"),
    "CXRMate": os.path.join(paths.GENERATIONS, "cxrmate_withprior.json"),
    "MLRG": os.path.join(paths.GENERATIONS, "mlrg_withprior.json"),
}
preds = {n: json.load(open(p)) for n, p in PREDS.items()}

CONFIGS = [
    ("official prior + official GT (paper baseline)", OFF_PRIOR, OFF_GT),
    ("NEURAL prior + official GT", CXB_PRIOR, OFF_GT),
    ("NEURAL prior + NEURAL GT (fully independent)", CXB_PRIOR, CXB_GT),
]

for title, prior, gt in CONFIGS:
    print("=" * 74)
    print(title)
    print("=" * 74)
    res = {}
    for name in ["PriorEdit", "sw", "CXRMate", "MLRG"]:
        rows, n = TA.score(preds[name], prior, gt)
        cp, cr, cf, cpred, ctrue, ctp = rows["change(micro)"]
        rp = rows["resolved"][0]
        res[name] = (cp, cr, rp, ctp, cpred)
        print(f"  {name:10s} change P={cp*100:4.0f}% R={cr*100:4.0f}%  resolved-P={rp*100:3.0f}%  "
              f"(change tp={ctp}/{cpred}, n_studies={n})")
    gap = (res['PriorEdit'][0] - res['sw'][0]) * 100
    print(f"  --> PriorEdit - sw change-precision gap = {gap:+.0f} points")
    print()
