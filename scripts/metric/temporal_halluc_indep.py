#!/usr/bin/env python
"""Temporal hallucination rate with an INDEPENDENT prior ground truth.

temporal_halluc_eval.py judges 'does the prior support this claim' with our OWN regex extract() -- the
same extractor the editor uses to GENERATE the claim, so a 0% is partly tautological (any extractor miss
is invisible to both sides). Here the prior-presence GT comes from the CheXbert labeler instead, which is
independent of our regex. A claim survives only if a DIFFERENT model agrees the prior supports it, so an
extractor miss on the prior (e.g. 'rib deformities' not matched as fracture -> false 'fracture new') is
now caught.

  present_req (unchanged/increased/decreased/resolved) on X -> requires CheXbert(prior)[X] present
  absent_req  (new/interval development)              on X -> requires CheXbert(prior)[X] absent

Claim detection on the PREDICTION is unchanged (regex; that legitimately measures what the report asserts).
present = CheXbert class in {1 positive, 3 uncertain} (uncertain->positive, same binarization as the paper
CE metric).

Run in env llmrg4:
  {PY_MODEL} temporal_halluc_indep.py --pred <json> --ann <ann> --name <label>
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import json, argparse, os, sys, collections
RG4 = paths.LLMRG4
os.chdir(RG4); sys.path.insert(0, RG4)
sys.path.insert(0, paths.SCRIPTS)
import numpy as np, torch
from models.chexbert import CheXbert
from temporal_halluc_eval import claims_with_type, FINDINGS

# our finding name -> CheXbert 14-head index (standard CheXpert order)
CHX_IDX = {"enlarged_cardiomediastinum": 0, "cardiomegaly": 1, "lung_opacity": 2, "lung_lesion": 3,
           "edema": 4, "consolidation": 5, "pneumonia": 6, "atelectasis": 7, "pneumothorax": 8,
           "pleural_effusion": 9, "pleural_other": 10, "fracture": 11, "support_devices": 12}


def chexbert_present(chx, reports, bs=64):
    """-> np.bool array [N, 14]; present iff class in {1 positive, 3 uncertain}."""
    out = []
    for i in range(0, len(reports), bs):
        arr = chx(list(reports[i:i + bs])).cpu().numpy()
        out.append(arr)
    arr = np.vstack(out)
    return (arr == 1) | (arr == 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--ann", required=True)
    ap.add_argument("--name", default="pred")
    a = ap.parse_args()
    pred = json.load(open(a.pred))
    priors = {}
    for r in json.load(open(a.ann))["test"]:
        lf = r.get("last_finding", "")
        pv = "" if lf.strip() == "s" else lf
        priors[str(r["id"])] = pv
        priors[str(r["study_id"])] = pv

    # label every prior we will need, once
    keys = [str(sid) for sid in pred if priors.get(str(sid), "").strip()]
    texts = [priors[k] for k in keys]
    dev = torch.device("cuda:0")
    chx = CheXbert(ckpt_dir="./hf", bert_path="./hf/bert-base-uncased",
                   checkpoint_path="./hf/chexbert.pth", device=dev).to(dev)
    present = chexbert_present(chx, texts)
    prior_pos = {k: present[i] for i, k in enumerate(keys)}   # k -> [14] bool

    total = halluc = 0
    by_type = collections.Counter()
    examples = []
    for sid, rep in pred.items():
        rep = rep[0] if isinstance(rep, list) else rep
        pp = prior_pos.get(str(sid))
        if pp is None:
            continue                          # no prior text -> can't judge
        for fd, types in claims_with_type(rep).items():
            if fd not in CHX_IDX:
                continue
            present_in_prior = bool(pp[CHX_IDX[fd]])
            for t in types:
                total += 1
                bad = (t == "present_req" and not present_in_prior) or \
                      (t == "absent_req" and present_in_prior)
                if bad:
                    halluc += 1
                    by_type[("present_req_but_prior_absent" if t == "present_req"
                             else "new_but_prior_present")] += 1
                    if len(examples) < 12:
                        examples.append((str(sid), fd, t, present_in_prior))
    rate = halluc / total * 100 if total else 0.0
    print(f"=== INDEPENDENT (CheXbert prior GT) temporal hallucination | {a.name} ===")
    print(f"  temporal claims checked: {total}")
    print(f"  hallucinated (CheXbert prior can't support): {halluc} ({rate:.1f}%)")
    print(f"  breakdown: {dict(by_type)}")
    for e in examples[:12]:
        print(f"    {e[0]} {e[1]:20s} claim={e[2]:12s} prior_present={e[3]}")


if __name__ == "__main__":
    main()
