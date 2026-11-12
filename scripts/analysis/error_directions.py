#!/usr/bin/env python
"""Error directions per model, per study and per finding (paper Table 4, App. H Tables 9 and 10).

On the 1337 with-prior studies, every asserted transition is compared with the true transition
(prior + current-GT CheXpert labels, exactly as in temporal_assessment.py):

  false "resolved"   asserted resolved,   true transition is not resolved
    all-clear        ... of which the true transition is persistent (ongoing disease called gone)
  false "unchanged"  asserted persistent, true transition is not persistent
    res.->unch.      ... of which the true transition is resolved
  new -> "unchanged" asserted persistent, true transition is new
  "new" when not new asserted new,        true transition is not new
  correct "resolved" asserted resolved,   true transition is resolved

Prints (1) Table 4 for PriorEdit and sw, (2) Table 9 for the four prior-aware models (total, mean per
study, share of studies with at least one error), (3) Table 10 per finding for PriorEdit and sw, and
(4) how the resolved findings called unchanged split by the current-GT label state
(unmentioned / uncertain / explicitly negated; paper: PriorEdit 339 / 77 / 43, sw 128 / 25 / 16).

Reads the shipped labels and generation files only; imports the shared temporal_assessment metric.
Env: bare Python 3.10+ (standard library); no GPU.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import json, os, collections
from temporal_assessment import FIND, asserted_transitions, true_transitions  # shared, unchanged

PRIOR = json.load(open(os.path.join(paths.LABELS, "prior_chexpert_labels.json")))
GT = json.load(open(os.path.join(paths.LABELS, "gt_chexpert_labels.json")))
MODELS = [
    ("PriorEdit", "prioredit.json"),
    ("sw", "llmrg4_sw.json"),
    ("CXRMate", "cxrmate_withprior.json"),
    ("MLRG", "mlrg_withprior.json"),
]
GT_STATE = {"none": "unmentioned", "unc": "uncertain", "neg": "negated"}


def load_pred(fn):
    d = json.load(open(os.path.join(paths.GENERATIONS, fn)))
    return {k: (v[0] if isinstance(v, list) else v) for k, v in d.items()}


def count(pred):
    """Per-study and per-finding error counts for one model."""
    ids = [i for i in PRIOR if i in GT and i in pred]
    per_study = []                       # (false resolved, false unchanged) per study
    tot = collections.Counter()
    by_f = {f: collections.Counter() for f in FIND}
    res_unch_gt = collections.Counter()  # current-GT label state of resolved findings called unchanged
    for i in ids:
        tt = true_transitions(PRIOR[i], GT[i])
        at = asserted_transitions(pred[i])
        fr = fu = 0
        for f in FIND:
            a, t = at[f], tt[f]
            c = by_f[f]
            if a == "resolved":
                if t == "resolved":
                    c["correct_res"] += 1
                else:
                    c["false_res"] += 1; fr += 1
                    if t == "persistent":
                        c["all_clear"] += 1
            elif a == "persistent" and t != "persistent":
                c["false_unch"] += 1; fu += 1
                if t == "resolved":
                    c["res_unch"] += 1
                    res_unch_gt[GT_STATE.get(GT[i].get(f, "none"), "other")] += 1
                elif t == "new":
                    c["new_unch"] += 1
            elif a == "new" and t != "new":
                c["false_new"] += 1
        per_study.append((fr, fu))
    for f in FIND:
        tot.update(by_f[f])
    return ids, per_study, tot, by_f, res_unch_gt


R = {name: count(load_pred(fn)) for name, fn in MODELS}

print("=" * 78)
print("Table 4  error types (finding instances, 1337 with-prior studies)")
print("=" * 78)
print(f"{'error type':38s}{'PriorEdit':>10}{'sw':>8}")
for label, key in [('persistent -> "resolved" (all-clear)', "all_clear"),
                   ('new -> "unchanged"', "new_unch"),
                   ('"new" when not new', "false_new"),
                   ('resolved -> "unchanged"', "res_unch"),
                   ('correct "resolved"', "correct_res")]:
    print(f"{label:38s}{R['PriorEdit'][2][key]:>10}{R['sw'][2][key]:>8}")
print()

print("=" * 78)
print("Table 9  both error directions per model")
print("=" * 78)
print(f"{'model':10s}{'n':>6} | {'fRes':>5}{'/study':>8}{'%st':>6}{'allclr':>7} | "
      f"{'fUnch':>6}{'/study':>8}{'%st':>6}{'res>unch':>9} | {'corrRes':>7}")
for name, _ in MODELS:
    ids, ps, tot, _, _ = R[name]
    n = len(ids)
    fr = [x for x, _ in ps]; fu = [y for _, y in ps]
    print(f"{name:10s}{n:>6} | {sum(fr):>5}{sum(fr)/n:>8.3f}{100*sum(x > 0 for x in fr)/n:>6.1f}"
          f"{tot['all_clear']:>7} | {sum(fu):>6}{sum(fu)/n:>8.3f}{100*sum(y > 0 for y in fu)/n:>6.1f}"
          f"{tot['res_unch']:>9} | {tot['correct_res']:>7}")
print()

print("=" * 78)
print("Table 10  per-finding error counts (PriorEdit | sw)")
print("=" * 78)
keys = ["false_res", "all_clear", "false_unch", "res_unch", "correct_res"]
hdr = "".join(f"{k:>12}" for k in ["F.res", "all-clear", "F.unch", "res>unch", "Corr."])
print(f"{'finding':28s}{hdr} |{hdr}")
for f in FIND:
    a = R["PriorEdit"][3][f]; b = R["sw"][3][f]
    print(f"{f:28s}" + "".join(f"{a[k]:>12}" for k in keys) + " |"
          + "".join(f"{b[k]:>12}" for k in keys))
a = R["PriorEdit"][2]; b = R["sw"][2]
print(f"{'Total':28s}" + "".join(f"{a[k]:>12}" for k in keys) + " |"
      + "".join(f"{b[k]:>12}" for k in keys))
print()

print("=" * 78)
print("Resolved findings called unchanged, by current-GT label state")
print("=" * 78)
for name in ("PriorEdit", "sw"):
    c = R[name][4]
    print(f"{name:10s} unmentioned={c['unmentioned']}  uncertain={c['uncertain']}  "
          f"negated={c['negated']}  (total {sum(c.values())})")
