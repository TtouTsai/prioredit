#!/usr/bin/env python
"""Score MS-CXR-T temporal progression predictions (matches the benchmark's macro-accuracy).

Metric: per finding, macro-accuracy = mean per-class recall over {improving, stable, worsening}
(balanced accuracy, handles class imbalance); then average over the 5 findings. SOTA reference:
CoCa-CXR 65.0%, BioViL-T 60.2% (both use the prior IMAGE; we are text-only-prior, a harder setting).

Predictions file: JSON {pair_id: {finding: "improving"|"stable"|"worsening"}} (or a --baseline).
Run:
  python score_mscxrt.py --pred preds.json
  python score_mscxrt.py --baseline majority     # sanity-check the harness / chance level
  python score_mscxrt.py --baseline always_stable
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import os, json, argparse, collections
MANIFEST = os.path.join(paths.OUT, "mscxrt/manifest.jsonl")
FINDINGS = ["consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"]
CLASSES = ["improving", "stable", "worsening"]


def load_gold():
    gold = {}  # pair_id -> {finding: class}
    for line in open(MANIFEST):
        r = json.loads(line); gold[r["pair_id"]] = r["labels"]
    return gold


def macro_acc(pairs, gold, pred):
    """per finding: mean recall over the 3 classes present; then average over findings."""
    per_finding = {}
    for fd in FINDINGS:
        # confusion: per true class, count correct
        by_true = collections.defaultdict(lambda: [0, 0])  # class -> [correct, total]
        for pid in pairs:
            g = gold[pid].get(fd)
            if g is None:
                continue
            p = pred.get(pid, {}).get(fd, None)
            by_true[g][1] += 1
            if p == g:
                by_true[g][0] += 1
        recalls = [c / t for c, t in by_true.values() if t]
        n = sum(t for _, t in by_true.values())
        per_finding[fd] = (sum(recalls) / len(recalls) if recalls else float("nan"), n)
    valid = [v for v, _ in per_finding.values() if v == v]
    overall = sum(valid) / len(valid) if valid else float("nan")
    return per_finding, overall


def build_baseline(kind, gold):
    pred = collections.defaultdict(dict)
    if kind == "always_stable":
        for pid, labs in gold.items():
            for fd in labs:
                pred[pid][fd] = "stable"
    elif kind == "majority":
        # majority class per finding
        maj = {}
        for fd in FINDINGS:
            c = collections.Counter(labs[fd] for labs in gold.values() if fd in labs)
            maj[fd] = c.most_common(1)[0][0] if c else "stable"
        for pid, labs in gold.items():
            for fd in labs:
                pred[pid][fd] = maj[fd]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred"); ap.add_argument("--baseline", choices=["majority", "always_stable"])
    a = ap.parse_args()
    gold = load_gold()
    pairs = list(gold)
    if a.baseline:
        pred = build_baseline(a.baseline, gold); name = f"baseline:{a.baseline}"
    else:
        pred = json.load(open(a.pred)); name = a.pred
    per_finding, overall = macro_acc(pairs, gold, pred)
    print(f"=== MS-CXR-T temporal progression | {name} ===")
    print(f"{'finding':16s}{'macro-acc':>11}{'n':>6}")
    for fd in FINDINGS:
        v, n = per_finding[fd]
        print(f"  {fd:14s}{v*100:>10.1f}%{n:>6}")
    print(f"  {'AVERAGE (5)':14s}{overall*100:>10.1f}%")
    print(f"\n  reference: BioViL-T 60.2% / CoCa-CXR 65.0% (both use prior IMAGE)")


if __name__ == "__main__":
    main()
