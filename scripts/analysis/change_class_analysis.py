#!/usr/bin/env python
"""Change-class analyses: Sec. 7 absolute-error accounting and the strict change classes (Sec. 7, App. E).

(a) Absolute-error accounting: total temporal claims / correct / wrong / per-report, for sw, PriorEdit
    and the baselines.
(b) Strict `new` analysis: new precision restricted to the subset where the prior EXPLICITLY negates the
    finding (labeler prior == neg), plus the composition of true-new (prior negated vs merely unmentioned).
(c) Strict `resolved` analysis: the truth restricted to resolutions the CURRENT GT report explicitly
    negates (labeler current == neg; 220 of 1356), with the composition of true-resolved, strict resolved
    precision (every asserted `resolved` outside that subset counts as an error) and strict recall.

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
import json, os, sys, collections
from temporal_assessment import FIND, asserted_transitions, true_transitions  # shared, unchanged

PRIOR = json.load(open(os.path.join(paths.LABELS, "prior_chexpert_labels.json")))
GT = json.load(open(os.path.join(paths.LABELS, "gt_chexpert_labels.json")))
PREDS = {
    "sw": os.path.join(paths.GENERATIONS, "llmrg4_sw.json"),
    "PriorEdit": os.path.join(paths.GENERATIONS, "prioredit.json"),
    "CXRMate": os.path.join(paths.GENERATIONS, "cxrmate_withprior.json"),
    "MLRG": os.path.join(paths.GENERATIONS, "mlrg_withprior.json"),
}


def load_pred(path):
    d = json.load(open(path))
    return {k: (v[0] if isinstance(v, list) else v) for k, v in d.items()}


def ids_for(pred):
    return [i for i in PRIOR if i in GT and i in pred]


def prior_state(i, f):
    """official CheXpert prior label state: 'pos' | 'neg' | 'unc' | 'none' (none = not mentioned)."""
    lab = PRIOR[i].get(f)
    return lab  # 'pos'|'neg'|'unc'|'none'


print("=" * 78)
print("(a)  Absolute-error accounting (change classes new+resolved + persistent)")
print("=" * 78)
n_report = None
for name in ["sw", "PriorEdit", "CXRMate", "MLRG"]:
    pred = load_pred(PREDS[name])
    ids = ids_for(pred)
    n_report = len(ids)
    tot = correct = 0
    per_class = collections.Counter()
    for i in ids:
        tt = true_transitions(PRIOR[i], GT[i])
        at = asserted_transitions(pred[i])
        for f in FIND:
            a = at[f]
            if a in ("new", "resolved", "persistent"):
                tot += 1
                per_class[("pred", a)] += 1
                if a == tt[f]:
                    correct += 1
                    per_class[("ok", a)] += 1
    wrong = tot - correct
    print(f"{name:10s} studies={len(ids)}  total_temporal_claims={tot}  correct={correct}  "
          f"wrong={wrong}  wrong/report={wrong/len(ids):.2f}")
print()

print("=" * 78)
print("(b)  'new'-class strict analysis")
print("=" * 78)
# composition of true-new: prior explicitly negated vs merely unmentioned
comp = collections.Counter()
sample_ids = ids_for(load_pred(PREDS["PriorEdit"]))
for i in sample_ids:
    tt = true_transitions(PRIOR[i], GT[i])
    for f in FIND:
        if tt[f] == "new":  # prior absent (neg or none) & current present
            comp[prior_state(i, f)] += 1
tot_new = sum(comp.values())
print(f"true-new composition (n={tot_new}):")
for st in ("neg", "none", "unc", "pos"):
    if comp[st]:
        print(f"  prior={st:5s}: {comp[st]:5d}  ({comp[st]/tot_new*100:.1f}%)")
print("  -> 'prior=none' are merely-unmentioned; 'prior=neg' are clinically-clean new.")
print()

# new precision on the STRICT subset: only count asserted-new where prior==neg (explicitly negated)
print("new precision: BROAD (all true-new) vs STRICT (prior explicitly-negated only)")
for name in ["sw", "PriorEdit", "CXRMate", "MLRG"]:
    pred = load_pred(PREDS[name])
    ids = ids_for(pred)
    # broad
    b_pred = b_tp = 0
    # strict: restrict the TRUTH to prior==neg; an asserted-new is a TP only if true-new AND prior==neg;
    # precision denominator = asserted-new whose prior==neg (the finding the model calls new was
    # explicitly negated in prior -> a clean 'new' claim), matching the strict-resolved treatment.
    s_pred = s_tp = 0
    for i in ids:
        tt = true_transitions(PRIOR[i], GT[i])
        at = asserted_transitions(pred[i])
        for f in FIND:
            if at[f] == "new":
                b_pred += 1
                if tt[f] == "new":
                    b_tp += 1
                if prior_state(i, f) == "neg":
                    s_pred += 1
                    if tt[f] == "new":
                        s_tp += 1
    bp = b_tp / b_pred if b_pred else 0
    sp = s_tp / s_pred if s_pred else 0
    print(f"  {name:10s} broad new-prec={bp*100:4.0f}% ({b_tp}/{b_pred})   "
          f"strict(prior=neg) new-prec={sp*100:4.0f}% ({s_tp}/{s_pred})")
print()

print("=" * 78)
print("(c)  'resolved'-class strict analysis")
print("=" * 78)
# composition of true-resolved: current GT explicitly negated vs merely unmentioned vs uncertain
comp = collections.Counter()
for i in sample_ids:
    tt = true_transitions(PRIOR[i], GT[i])
    for f in FIND:
        if tt[f] == "resolved":  # prior present & current not present
            comp[GT[i].get(f, "none")] += 1
tot_res = sum(comp.values())
print(f"true-resolved composition (n={tot_res}):")
for st in ("neg", "none", "unc"):
    if comp[st]:
        print(f"  current={st:5s}: {comp[st]:5d}  ({comp[st]/tot_res*100:.1f}%)")
print("  -> 'current=neg' are the clinically unambiguous (explicitly negated) resolutions.")
print()

print("resolved precision: BROAD vs STRICT (current explicitly negated), and strict recall")
for name in ["sw", "PriorEdit", "CXRMate", "MLRG"]:
    pred = load_pred(PREDS[name])
    ids = ids_for(pred)
    n_pred = b_tp = s_tp = 0
    for i in ids:
        tt = true_transitions(PRIOR[i], GT[i])
        at = asserted_transitions(pred[i])
        for f in FIND:
            if at[f] == "resolved":
                n_pred += 1
                if tt[f] == "resolved":
                    b_tp += 1
                    if GT[i].get(f) == "neg":
                        s_tp += 1
    bp = b_tp / n_pred if n_pred else 0
    sp = s_tp / n_pred if n_pred else 0
    print(f"  {name:10s} broad resolved-prec={bp*100:4.0f}% ({b_tp}/{n_pred})   "
          f"strict resolved-prec={sp*100:4.0f}% ({s_tp}/{n_pred})   "
          f"strict recall={s_tp}/{comp['neg']}")
