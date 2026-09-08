#!/usr/bin/env python
"""GT-based temporal accuracy (direction-aware). Unlike temporal_halluc_* (which only checks a claim
against the PRIOR = grounding), this checks each temporal claim against the TRUE transition derived from
(prior state, CURRENT-GT state), both from the official CheXpert labeler. This tests whether the
temporal claims are CORRECT, not just prior-safe -- and it is NOT circular: the report's "current" comes
from the image-only draft, the GT's "current" from the real report.

TRUE transition per finding, from prior P and current-GT G (present = labeler 'pos'):
  P absent, G present -> new ; P present, G absent -> resolved ; P present, G present -> persistent ;
  else -> none (absent both times, no temporal).
ASSERTED transition per finding: parse the generated report's comparison words bound to the finding
(increased/decreased/unchanged all collapse to 'persistent' since both-present).

Reports per-category precision / recall / F1, plus a micro over the CHANGE classes {new, resolved}
(the informative events; persistent is abundant and less diagnostic).

  python temporal_accuracy.py --prior <prior official labels> --gt <gt official labels> \
      --preds name1:<report json> name2:<report json> ...
"""
import os as _os, sys as _sys
_scripts = _os.path.dirname(_os.path.abspath(__file__))
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402,F401  (also puts scripts/* groups on sys.path)
import json, re, os, sys, argparse, collections
from state_extractor import extract, FIND_KW

FIND = [f for f in FIND_KW if f != "support_devices"]

# temporal-word classifiers (whole sentence). Change classes take priority over persistent.
NEW = re.compile(r"\b(new|newly|interval development|developed|develop|appear\w*|now (?:seen|present|noted|evident)|has developed)\b", re.I)
RESOLVED = re.compile(r"\b(resol\w*|cleared|clearing|no longer|removed|removal|reaccumulat\w*)\b", re.I)
# NOTE: 'enlarg*' (enlarged/enlargement) describes a static SIZE STATE (e.g. 'cardiac silhouette is
# enlarged' = cardiomegaly), not a comparison to the prior, so it is NOT a persistence word. Kept
# words all imply an explicit comparison (stability or a size/severity change).
PERSIST = re.compile(r"\b(unchanged|stable|stably|persist\w*|constant|unaltered|again|redemonstrat\w*|previously|increas\w*|decreas\w*|improv\w*|worsen\w*|larger|smaller|reduced|progress\w*|similar)\b", re.I)
NO_NEW = re.compile(r"\bno (?:new|newly|interval|significant interval|evidence of new)|without (?:new|interval)", re.I)
NOT_RESOLVED = re.compile(r"\b(?:unresolved|not (?:fully |completely )?resolved|partially resolved|incompletely resolved)\b", re.I)


def asserted_transitions(report):
    """Per-finding asserted transition. Binding is polarity-aware so negated temporal does not fire:
    'new' needs the finding stated POSITIVE (so 'no new effusion' -> effusion neg -> not new);
    'resolved' fires on the resolved-word (guarded against 'not resolved'); 'persistent' needs POS."""
    out = {f: "none" for f in FIND}
    for s in re.split(r"[.;\n]", (report or "")):
        s_new = bool(NEW.search(s)) and not NO_NEW.search(s)
        s_res = bool(RESOLVED.search(s)) and not NOT_RESOLVED.search(s)
        s_per = bool(PERSIST.search(s))
        if not (s_new or s_res or s_per):
            continue
        ext = extract(s)
        for f in FIND:
            st = ext[f][0]
            if st == "none":
                continue
            if s_new and st == "pos":                       # genuinely new -> present now
                out[f] = "new"
            elif s_res:                                      # resolved-word dominates (finding now gone)
                out[f] = "resolved"
            elif s_per and st == "pos" and out[f] == "none":  # unchanged/increased/decreased -> present both
                out[f] = "persistent"
    return out


def true_transitions(prior_lab, gt_lab):
    out = {}
    for f in FIND:
        pp = prior_lab.get(f) == "pos"
        gg = gt_lab.get(f) == "pos"
        if pp and gg:
            out[f] = "persistent"
        elif not pp and gg:
            out[f] = "new"
        elif pp and not gg:
            out[f] = "resolved"
        else:
            out[f] = "none"
    return out


def score(preds, prior, gt):
    ids = [i for i in prior if i in gt and i in preds]
    cats = ["new", "resolved", "persistent"]
    tp = collections.Counter(); n_pred = collections.Counter(); n_true = collections.Counter()
    for i in ids:
        tt = true_transitions(prior[i], gt[i])
        at = asserted_transitions(preds[i][0] if isinstance(preds[i], list) else preds[i])
        for f in FIND:
            t, a = tt[f], at[f]
            if a in cats:
                n_pred[a] += 1
                if a == t:
                    tp[a] += 1
            if t in cats:
                n_true[t] += 1
    rows = {}
    for c in cats:
        p = tp[c] / n_pred[c] if n_pred[c] else 0.0
        r = tp[c] / n_true[c] if n_true[c] else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows[c] = (p, r, f1, n_pred[c], n_true[c], tp[c])
    # micro over change classes {new, resolved}
    ctp = tp["new"] + tp["resolved"]; cpred = n_pred["new"] + n_pred["resolved"]; ctrue = n_true["new"] + n_true["resolved"]
    cp = ctp / cpred if cpred else 0.0; cr = ctp / ctrue if ctrue else 0.0
    cf = 2 * cp * cr / (cp + cr) if (cp + cr) else 0.0
    rows["change(micro)"] = (cp, cr, cf, cpred, ctrue, ctp)
    return rows, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", nargs="+", required=True, help="name:report_json ...")
    a = ap.parse_args()
    prior = json.load(open(a.prior)); gt = json.load(open(a.gt))
    print(f"{'method':10s}{'category':16s}{'P':>7}{'R':>7}{'F1':>7}{'#pred':>7}{'#true':>7}{'#tp':>6}")
    for spec in a.preds:
        name, path = spec.split(":", 1)
        preds = json.load(open(path))
        rows, n = score(preds, prior, gt)
        for c in ["new", "resolved", "persistent", "change(micro)"]:
            p, r, f1, npd, ntr, tp = rows[c]
            print(f"{name:10s}{c:16s}{p*100:>6.0f}%{r*100:>6.0f}%{f1*100:>6.0f}%{npd:>7}{ntr:>7}{tp:>6}")
        print()


if __name__ == "__main__":
    main()
