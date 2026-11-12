#!/usr/bin/env python
"""Study-level bootstrap 95% CIs for presence (CheXbert-14 micro-F1) and the system differences (paper
Table 2, §7 and App. C). Labels each system's generated report and its ground truth with neural CheXbert, forms
a per-study 14-vector, and resamples studies (B=2000) to get each system's F1 CI and the paired
PriorEdit-minus-baseline difference CI.

Two GT conventions, reported separately in the paper:
  --canonical  scored against the LLM-RG4 clean GT: PriorEdit vs sw (+0.22, CI [-0.79, +1.23], parity)
               and vs its own sn base (+1.11, CI [+0.69, +1.52], a real lift)
  --orig       scored against the original MIMIC-CXR GT: PriorEdit trails sw (-1.71, CI [-2.75, -0.69])

Each argument is a whitespace-separated list of name:eval_dir, where eval_dir holds pred.csv and gt.csv
with a "Report Impression" column (produced by scripts/metric/eval_stage2_radeval.py from the shipped
generations and your GT text). The first name in each split is the reference (PriorEdit); differences
are taken against it. GT report text is MIMIC-CXR under a PhysioNet DUA and is not redistributed here.

Env: needs the CheXbert checkpoint and a GPU (imports models.chexbert from the LLM-RG4
checkout); the labeling/metric env suffices -- the full model-inference stack is not required.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots from PRIOREDIT_* env vars)
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch

RG = paths.LLMRG4
sys.path.insert(0, RG)
os.chdir(RG)  # models.chexbert uses relative ./hf paths
from models.chexbert import CheXbert  # noqa: E402

_CHX = None


def chx():
    global _CHX
    if _CHX is None:
        dev = torch.device("cuda:0")
        _CHX = CheXbert(ckpt_dir="./hf", bert_path="./hf/bert-base-uncased",
                        checkpoint_path="./hf/chexbert.pth", device=dev).to(dev)
    return _CHX


def present(reports, bs=64):
    """Neural CheXbert -> (n, 14) in {0,1}. raw class: 1=pos, 3=uncertain(->pos), 2=neg, 0=blank."""
    out = []
    for i in range(0, len(reports), bs):
        out.append(chx()(list(reports[i:i + bs])).cpu().numpy())
    arr = np.vstack(out) if out else np.zeros((0, 14))
    df = pd.DataFrame(arr).replace(0, np.nan).replace(3, -1).replace(2, 0).fillna(0)
    a = df.to_numpy().astype(float)
    a[a == -1] = 1
    return a


def read_col(path):
    return pd.read_csv(path)["Report Impression"].fillna("").astype(str).tolist()


def label_dir(d):
    """Label an eval dir's pred.csv/gt.csv -> (gt_vectors, pred_vectors), row-aligned per study."""
    gt = read_col(os.path.join(d, "gt.csv"))
    pred = read_col(os.path.join(d, "pred.csv"))
    n = min(len(gt), len(pred))
    return present(gt[:n]), present(pred[:n])


def micro_f1(g, p, idx):
    G, P = g[idx], p[idx]
    tp = np.logical_and(G == 1, P == 1).sum()
    fp = np.logical_and(G == 0, P == 1).sum()
    fn = np.logical_and(G == 1, P == 0).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def pctile_ci(a):
    a = np.sort(a)
    return round(np.percentile(a, 2.5) * 100, 2), round(np.percentile(a, 97.5) * 100, 2)


def bootstrap_split(title, specs, B, seed=0):
    labels = {name: label_dir(d) for name, d in specs}
    names = [n for n, _ in specs]
    ref = names[0]
    n = min(len(labels[k][0]) for k in labels)
    base = np.arange(n)
    point = {k: micro_f1(*labels[k], base) for k in labels}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in labels}
    diff = {k: [] for k in names[1:]}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        f = {k: micro_f1(*labels[k], idx) for k in labels}
        for k in labels:
            boot[k].append(f[k])
        for k in names[1:]:
            diff[k].append(f[ref] - f[k])
    print(f"\n=== {title}  (n={n}, B={B}) ===")
    for k in names:
        lo, hi = pctile_ci(boot[k])
        print(f"  {k:10s} F1 {point[k]*100:6.2f}  CI[{lo:.2f}, {hi:.2f}]")
    for k in names[1:]:
        d = (point[ref] - point[k]) * 100
        lo, hi = pctile_ci(diff[k])
        sig = "" if lo <= 0 <= hi else "  (significant)"
        print(f"  {ref}-{k} diff {d:+.2f}  CI[{lo:+.2f}, {hi:+.2f}]{sig}")


def parse_specs(items):
    return [(s.split(":", 1)[0], s.split(":", 1)[1]) for s in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", nargs="+", metavar="name:dir",
                    help="LLM-RG4 clean-GT eval dirs, reference first (e.g. PriorEdit:... sw:... sn:...)")
    ap.add_argument("--orig", nargs="+", metavar="name:dir",
                    help="original-MIMIC-GT eval dirs, reference first (e.g. PriorEdit:... sw:...)")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    if not (a.canonical or a.orig):
        ap.error("give at least one of --canonical / --orig")
    if a.canonical:
        bootstrap_split("canonical (LLM-RG4 clean GT)", parse_specs(a.canonical), a.B)
    if a.orig:
        bootstrap_split("original MIMIC-CXR GT", parse_specs(a.orig), a.B)


if __name__ == "__main__":
    main()
