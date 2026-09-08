#!/usr/bin/env python
"""Paper-aligned CheXbert CE metric: reuses LLM-RG4's own CheXbert labeler + compute_mlc + binarization
(uncertain->positive, negative/blank->0), so the numbers use the SAME implementation as the LLM-RG4
paper. Reports CheXbert-14 micro F1/Precision/Recall (the paper's CE) plus macro.

Run in env llmrg4:
  {PY_MODEL} paper_chexbert_eval.py <eval_dir> [<eval_dir> ...]
Each eval_dir has pred.csv/gt.csv with a 'Report Impression' column.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import sys, os
RG4 = paths.LLMRG4
os.chdir(RG4); sys.path.insert(0, RG4)
import numpy as np, pandas as pd, torch
from models.chexbert import CheXbert
from models.metrics import compute_mlc


def label(chexbert, reports, bs=64):
    out = []
    for i in range(0, len(reports), bs):
        out.append(chexbert(list(reports[i:i + bs])).cpu().numpy())
    return np.vstack(out)


def binarize(arr):
    df = pd.DataFrame(arr).replace(0, np.nan).replace(3, -1).replace(2, 0).fillna(0)
    a = df.to_numpy().astype(float); a[a == -1] = 1
    return a


def main():
    dirs = sys.argv[1:]
    dev = torch.device("cuda:0")
    chx = CheXbert(ckpt_dir="./hf", bert_path="./hf/bert-base-uncased",
                   checkpoint_path="./hf/chexbert.pth", device=dev).to(dev)
    rows = []
    for d in dirs:
        pred = pd.read_csv(f"{d}/pred.csv")["Report Impression"].fillna("").astype(str).tolist()
        gt = pd.read_csv(f"{d}/gt.csv")["Report Impression"].fillna("").astype(str).tolist()
        n = min(len(pred), len(gt)); pred, gt = pred[:n], gt[:n]
        g = binarize(label(chx, gt)); p = binarize(label(chx, pred))
        m = compute_mlc(g, p, [f"f{i}" for i in range(14)])
        rows.append((os.path.basename(d), n, m["F1_MICRO"], m["PRECISION_MICRO"], m["RECALL_MICRO"],
                     m["F1_MACRO"], m["PRECISION_MACRO"], m["RECALL_MACRO"]))
    print(f"\n{'method':22s}{'n':>6}{'F1_mic':>9}{'P_mic':>8}{'R_mic':>8}{'F1_mac':>9}{'P_mac':>8}{'R_mac':>8}")
    for r in rows:
        print(f"{r[0]:22s}{r[1]:>6}{r[2]*100:>8.2f}{r[3]*100:>8.2f}{r[4]*100:>8.2f}{r[5]*100:>9.2f}{r[6]*100:>8.2f}{r[7]*100:>8.2f}")


if __name__ == "__main__":
    main()
