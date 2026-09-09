#!/usr/bin/env python
"""Relabel the prior and current-GT reports with neural CheXbert -> present/absent per finding.

Produces the two label files the independent-labeler robustness check (independent_labeler.py) and
the noise-floor analysis (official-vs-neural agreement, paper 92.9%) consume:

  $PRIOREDIT_NEURAL_LABELS/chexbert_prior_labels.json   prior report (last_finding) state, per study id
  $PRIOREDIT_NEURAL_LABELS/chexbert_gt_labels.json      current-GT report (finding) state, per study id

(default dir: $PRIOREDIT_OUT/neural_labels). Each value is {finding: "pos"|"none"} -- present-status
only, no report text -- keyed by study id, so the output is safe to keep, but the input report text is
MIMIC-CXR under a PhysioNet DUA and is not redistributed here.

Reuses LLM-RG4's own CheXbert wrapper (same implementation as the presence metric). The neural labeler
is deliberately a *different* labeler from the official CheXpert rule labeler that supplies TCA's
states, so agreement between the two bounds the metric's label noise.

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

COND2NAME = ["enlarged_cardiomediastinum", "cardiomegaly", "lung_opacity", "lung_lesion", "edema",
             "consolidation", "pneumonia", "atelectasis", "pneumothorax", "pleural_effusion",
             "pleural_other", "fracture", "support_devices"]  # idx 13 = "No Finding", skipped


def clean(s):
    s = (s or "").strip()
    if s.lower() == "s":  # LLM-RG4's empty-prior sentinel
        return ""
    return s.replace("finding :", "").replace("finding:", "").strip()


def present(chx, reports, bs=64):
    """Neural CheXbert -> (n, 14) in {0,1}. raw class: 1=pos, 3=uncertain(->pos), 2=neg, 0=blank."""
    out = []
    for i in range(0, len(reports), bs):
        out.append(chx(list(reports[i:i + bs])).cpu().numpy())
    arr = np.vstack(out) if out else np.zeros((0, 14))
    df = pd.DataFrame(arr).replace(0, np.nan).replace(3, -1).replace(2, 0).fillna(0)
    a = df.to_numpy().astype(float)
    a[a == -1] = 1
    return a


def to_states(mat, ids):
    return {did: {COND2NAME[j]: ("pos" if mat[i, j] == 1 else "none") for j in range(13)}
            for i, did in enumerate(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=f"{RG}/data/swcmp/annotation.json",
                    help="LLM-RG4 sw test annotation (prior report = last_finding, current GT = finding)")
    ap.add_argument("--out", default=os.environ.get("PRIOREDIT_NEURAL_LABELS",
                    os.path.join(paths.OUT, "neural_labels")))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ann = json.load(open(a.ann))["test"]
    ids = [str(t["id"]) for t in ann]
    prior_txt = [clean(t.get("last_finding", "")) for t in ann]
    gt_txt = [clean(t.get("finding", "")) for t in ann]
    print(f"labeling {len(ids)} prior + {len(ids)} current-GT reports with neural CheXbert...", file=sys.stderr)

    dev = torch.device("cuda:0")
    chx = CheXbert(ckpt_dir="./hf", bert_path="./hf/bert-base-uncased",
                   checkpoint_path="./hf/chexbert.pth", device=dev).to(dev)
    prior_states = to_states(present(chx, prior_txt), ids)
    gt_states = to_states(present(chx, gt_txt), ids)

    json.dump(prior_states, open(os.path.join(a.out, "chexbert_prior_labels.json"), "w"))
    json.dump(gt_states, open(os.path.join(a.out, "chexbert_gt_labels.json"), "w"))
    print(f"wrote chexbert_prior_labels.json + chexbert_gt_labels.json ({len(ids)} studies) -> {a.out}")


if __name__ == "__main__":
    main()
