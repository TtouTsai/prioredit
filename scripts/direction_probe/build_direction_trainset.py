#!/usr/bin/env python
"""Build a SILVER direction-classification training set from the RG4 sw train split.

Each instance: for one of the 5 MS-CXR-T findings present with a comparison phrase in the CURRENT report,
  input  = (prior finding-context sentences from last_finding, current finding-context from finding)
  label  = improving | stable | worsening   (from the current report's comparison language)

Two guards keep this honest as a stand-in for the deployment setting (prior TEXT + image-only draft, which
has NO comparison words):
  1. exclude every MS-CXR-T *subject* (the test benchmark) -> no leakage.
  2. emit a MASKED copy of each instance with the comparison words stripped from the current context, so
     the model is forced to learn from the severity/presence delta (what the draft actually carries) and
     cannot just copy a direction word. Trained on original+masked mix.

Out: JSONL {finding, prior_ctx, current_ctx, label, masked, subject_id}. Split by subject into train/val.
Run: python build_direction_trainset.py --out_dir data/direction
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import json, re, argparse, os, random, collections

SW = os.path.join(paths.DATA, "mimic_rg4/sw_single_view_with_prior.json")
MANIFEST = os.path.join(paths.OUT, "mscxrt/manifest.jsonl")
FINDINGS = ["consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"]
FIND_KW = {
    "consolidation": [r"consolidat\w*"],
    "edema": [r"edema", r"vascular congestion", r"pulmonary congestion", r"fluid overload", r"\bchf\b"],
    "pleural_effusion": [r"pleural effusion", r"effusion", r"pleural fluid"],
    "pneumonia": [r"pneumonia", r"infectious", r"infection"],
    "pneumothorax": [r"pneumothorax", r"ptx"],
}
# comparison language -> direction (severity/interval change words). Descriptive-only size words
# (large/small/moderate) are NOT here: they describe state, direction needs an interval word.
WORSE = re.compile(r"\b(increased|increasing|increase|worsen\w*|enlarg\w*|progress\w*|newly|"
                   r"develop\w*|development|interval development|reaccumulat\w*|reappear\w*|"
                   r"more extensive|more prominent|greater|larger|advancing|expanding)\b")
BETTER = re.compile(r"\b(decreased|decreasing|decrease|improv\w*|reduced|reduction|resolv\w*|"
                    r"resolution|smaller|cleared|clearing|resorb\w*|reexpans\w*|re-expans\w*|"
                    r"less extensive|less prominent|near resolution|interval decrease|diminish\w*)\b")
STABLE = re.compile(r"\b(unchanged|stable|stably|persist\w*|unaltered|no change|no significant change|"
                    r"no interval change|not significantly changed|similar|redemonstrat\w*)\b")
NEW = re.compile(r"\bnew\b")
NO_NEW = re.compile(r"\b(no new|without new|no interval development|no evidence of new)\b")
ALLCH = re.compile("|".join([WORSE.pattern, BETTER.pattern, STABLE.pattern, r"\bnew\b"]))


def sents(t):
    return [s.strip() for s in re.split(r"[.;\n]", (t or "").lower()) if s.strip()]


def ctx(text, kws):
    return [s for s in sents(text) if any(re.search(k, s) for k in kws)]


def label_sentence(s):
    """direction from one current-report sentence, with a 'no new' negation guard. None if none/ambiguous."""
    no_new = bool(NO_NEW.search(s))
    w = bool(WORSE.search(s)) or (bool(NEW.search(s)) and not no_new)
    b = bool(BETTER.search(s))
    st = bool(STABLE.search(s))
    if w and not b:
        return "worsening"
    if b and not w:
        return "improving"
    if st and not (w or b):
        return "stable"
    return None


def label_finding(cur_sents):
    labs = [label_sentence(s) for s in cur_sents]
    labs = [x for x in labs if x]
    if not labs:
        return None
    u = set(labs)
    if len(u) == 1:
        return labs[0]
    # mixed sentences for the same finding -> ambiguous, drop
    return None


def mask_dir(text):
    return re.sub(r"\s+", " ", ALLCH.sub(" ", text)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(paths.OUT, "direction"))
    ap.add_argument("--val_frac", type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ms_sub = set(str(json.loads(l)["subject_id"]) for l in open(MANIFEST))
    d = json.load(open(SW))["train"]
    rows = []
    stat = collections.Counter()
    for r in d:
        if str(r["subject_id"]) in ms_sub:
            continue
        prior = r.get("last_finding", "")
        if prior.strip() in ("", "s"):
            continue
        cur = r.get("finding", "")
        for fd in FINDINGS:
            cs = ctx(cur, FIND_KW[fd])
            if not cs:
                continue
            lab = label_finding(cs)
            if lab is None:
                continue
            ps = ctx(prior, FIND_KW[fd])
            prior_ctx = " . ".join(ps)
            cur_ctx = " . ".join(cs)
            base = {"finding": fd, "prior_ctx": prior_ctx, "label": lab,
                    "subject_id": str(r["subject_id"])}
            rows.append({**base, "current_ctx": cur_ctx, "masked": 0})
            rows.append({**base, "current_ctx": mask_dir(cur_ctx), "masked": 1})
            stat[lab] += 1
    # split by subject
    subs = sorted(set(x["subject_id"] for x in rows))
    random.Random(0).shuffle(subs)
    n_val = int(len(subs) * a.val_frac)
    val_subs = set(subs[:n_val])
    tr = [x for x in rows if x["subject_id"] not in val_subs]
    va = [x for x in rows if x["subject_id"] in val_subs]
    with open(f"{a.out_dir}/train.jsonl", "w") as f:
        for x in tr:
            f.write(json.dumps(x) + "\n")
    with open(f"{a.out_dir}/val.jsonl", "w") as f:
        for x in va:
            f.write(json.dumps(x) + "\n")
    print(f"instances (x2 for masked): labels {dict(stat)}  total_rows {len(rows)}")
    print(f"train {len(tr)}  val {len(va)}  (val subjects {len(val_subs)})")
    print(f"-> {a.out_dir}/train.jsonl , val.jsonl")


if __name__ == "__main__":
    main()
