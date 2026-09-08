#!/usr/bin/env python
"""Unified, citable RRG metrics via RadEval (shared env `metriceval`, py3.11, torch cu128).
Reads pred.csv/gt.csv from our eval dirs and reports CheXbert micro/macro-14/5, RaTEScore, and
TemporalEntityF1 — same-scale comparison to SOTA (EditGRPO). RadGraph currently skipped (vendored
import bug in radeval 2.2.1). Run: PYTHONSAFEPATH=1 metriceval/bin/python radeval_score.py <eval_dir> ...
Each eval_dir must contain pred.csv and gt.csv with a 'Report Impression' column."""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import os, sys, json
os.environ.setdefault("PYTHONSAFEPATH", "1")
import pandas as pd
from radeval import RadEval

RES = os.path.join(paths.DATA, "RRG-DPO/results")
METRICS = ["f1chexbert", "ratescore", "temporal"]


def load(d, f):
    p = f"{RES}/{d}_eval/{f}.csv" if not os.path.isdir(d) else f"{d}/{f}.csv"
    return pd.read_csv(p)["Report Impression"].fillna("").astype(str).tolist()


def main():
    dirs = sys.argv[1:]
    ev = RadEval(metrics=METRICS)
    rows = []
    for d in dirs:
        hyps = load(d, "pred"); refs = load(d, "gt")
        n = min(len(hyps), len(refs)); hyps, refs = hyps[:n], refs[:n]
        out = ev(refs=refs, hyps=hyps)
        out["_name"] = d; out["_n"] = n
        rows.append(out)
        print(f"[{d}] n={n}  "
              f"CheXbert-14: micro {out.get('f1chexbert_all_micro_f1',0):.4f} macro {out.get('f1chexbert_all_macro_f1',0):.4f} | "
              f"CheXbert-5: micro {out.get('f1chexbert_5_micro_f1',0):.4f} macro {out.get('f1chexbert_5_macro_f1',0):.4f} | "
              f"RaTE {out.get('ratescore',0):.4f} | Temporal-F1 {out.get('temporal_f1',0):.4f}", flush=True)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rows, open(os.path.join(out_dir, "radeval_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
