#!/usr/bin/env python
"""Build an eval dir (pred.csv / gt.csv with a 'Report Impression' column) from a generation file,
for paper_chexbert_eval.py (presence) and radeval_score.py (RaTE).

The GT side is the current report's findings from the LLM-RG4 swcmp annotation, so the eval dir
contains MIMIC-CXR report text: keep it local (the PhysioNet DUA forbids redistribution).

  python make_eval_dir.py --pred data/generations/qwen3vl_imageonly.json \
      --ann $PRIOREDIT_LLMRG4/data/swcmp/annotation.json --out out/eval/qwen3vl --with_prior

--with_prior keeps only the 1337 with-prior studies (those in data/labels/), the cohort of Table 8.
Rows follow the generation file's key order. Env: bare Python 3.10+ (standard library).
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import os, csv, json, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="generation json {id: report}")
    ap.add_argument("--ann", required=True, help="LLM-RG4 swcmp annotation.json")
    ap.add_argument("--out", required=True, help="eval dir to create")
    ap.add_argument("--with_prior", action="store_true", help="restrict to the 1337 with-prior studies")
    a = ap.parse_args()
    ann = {str(r["id"]): r for r in json.load(open(a.ann))["test"]}
    pred = json.load(open(a.pred))
    keep = None
    if a.with_prior:
        prior = json.load(open(os.path.join(paths.LABELS, "prior_chexpert_labels.json")))
        gt = json.load(open(os.path.join(paths.LABELS, "gt_chexpert_labels.json")))
        keep = set(prior) & set(gt)
    rows_p, rows_g = [], []
    for sid, v in pred.items():
        if sid not in ann or (keep is not None and sid not in keep):
            continue
        rows_p.append(v[0] if isinstance(v, list) else v)
        g = ann[sid]["finding"]
        rows_g.append(g[0] if isinstance(g, list) else g)
    os.makedirs(a.out, exist_ok=True)
    for fn, rows in (("pred.csv", rows_p), ("gt.csv", rows_g)):
        with open(os.path.join(a.out, fn), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Report Impression"])
            for t in rows:
                w.writerow([t])
    print(f"wrote {a.out}/pred.csv & gt.csv  (n={len(rows_p)})")


if __name__ == "__main__":
    main()
