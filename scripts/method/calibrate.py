#!/usr/bin/env python
"""Per-finding confidence calibration (paper App. C, Table 6) -> configs/confidence_thresholds.json.

For each candidate finding, fit a logistic  P(y=1 | c) = sigma(a_f * c + b_f)  on the Eq. 2 coverage
scope of the sw VALIDATION split -- prior present AND the image-only draft silent -- then invert it
(Eq. 3) to the operating thresholds:

    tau_pos = (logit(base_f)   - b_f) / a_f      adopted positive-coverage threshold (P* = base_f)
    tau_neg = (logit(base_f/2) - b_f) / a_f      resolved (negative) branch        (P* = base_f/2)

and also the alternative P*=0.5 point reported (not adopted) in App. F. The label y is the
current-presence truth from neural CheXbert over the current-GT report text. Prints the Table 6 view
(n, base, a_f, b_f, tau_pos, tau_neg) and writes the adopted thresholds config.

Inputs are the VALIDATION split and are NOT redistributed (regenerate under your own PhysioNet DUA):
  PRIOREDIT_VAL_CONF   confidence c_f(I) per val study    (default: $PRIOREDIT_LLMRG4/save/val_prior_conf/result/finding_confidence.json)
                       produce with scripts/method/probe_confidence.py on the val split
  PRIOREDIT_VAL_DRAFT  image-only sn draft per val study  (default: $PRIOREDIT_LLMRG4/save/val_prior_sn/result/test_result.json)
  PRIOREDIT_VAL_ANN    val annotation with prior (last_finding) + current-GT (finding) report text,
                       as {"test": [{"id","last_finding","finding"}, ...]} -- MIMIC-CXR text under DUA
                       (default: $PRIOREDIT_LLMRG4/data/swcmp/val_prior_ann.json)

Env: needs the CheXbert checkpoint and a GPU (imports models.chexbert from the LLM-RG4
checkout); the labeling/metric env suffices -- the full model-inference stack is not required.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots from PRIOREDIT_* env vars; also puts scripts/* groups on sys.path)
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch

RG = paths.LLMRG4
sys.path.insert(0, RG)
os.chdir(RG)  # models.chexbert uses relative ./hf paths
from models.chexbert import CheXbert  # noqa: E402
from state_extractor import extract   # noqa: E402  (prior/draft state; ext[f][0] in {pos,neg,none})

# CheXbert CONDITIONS order (idx 0..12; 13 = "No Finding", skipped) -> our finding names.
COND2NAME = ["enlarged_cardiomediastinum", "cardiomegaly", "lung_opacity", "lung_lesion", "edema",
             "consolidation", "pneumonia", "atelectasis", "pneumothorax", "pleural_effusion",
             "pleural_other", "fracture", "support_devices"]

# Adopted design (App. C): a finding is opened for positive coverage when its fit is usable and its
# in-scope validation set is large enough and monotone; the resolved (negative) branch is opened only
# where a resolution is clinically meaningful. Findings with too few clean in-scope pairs (MIN_PAIRS)
# are left out of the config entirely (pleural effusion, consolidation).
MIN_PAIRS = 20
OPEN_POS = {"cardiomegaly", "lung_opacity", "edema", "pneumonia", "support_devices"}
OPEN_NEG = {"lung_opacity", "edema", "support_devices"}


def load_json(path, what):
    if not os.path.exists(path):
        sys.exit(f"missing {what}: {path}\n  (validation-split input; see this script's docstring / docs/DATA.md)")
    return json.load(open(path))


def first(v):
    return v[0] if isinstance(v, list) else v


def chexbert_present(chx, reports, bs=64):
    """Neural CheXbert -> (n, 14) present/absent in {0,1}. raw: 1=pos, 3=uncertain(->pos), 2=neg, 0=blank."""
    out = []
    for i in range(0, len(reports), bs):
        out.append(chx(list(reports[i:i + bs])).cpu().numpy())
    arr = np.vstack(out) if out else np.zeros((0, 14))
    df = pd.DataFrame(arr).replace(0, np.nan).replace(3, -1).replace(2, 0).fillna(0)
    a = df.to_numpy().astype(float)
    a[a == -1] = 1
    return a


def fit_logistic(x, y, iters=800, lr=0.02):
    """Standardized-space logistic fit, returned in original c-space: sigma(a_c*c + b_c)."""
    mu, sd = x.mean(), x.std() + 1e-6
    xs = (x - mu) / sd
    a, b = 0.0, 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * xs + b)))
        a -= lr * ((p - y) * xs).mean() * 10
        b -= lr * (p - y).mean() * 10
    return a / sd, b - a * mu / sd  # a_c, b_c


def tau_at(a_c, b_c, pstar):
    return (np.log(pstar / (1.0 - pstar)) - b_c) / a_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default=os.environ.get(
        "PRIOREDIT_VAL_CONF", f"{RG}/save/val_prior_conf/result/finding_confidence.json"))
    ap.add_argument("--draft", default=os.environ.get(
        "PRIOREDIT_VAL_DRAFT", f"{RG}/save/val_prior_sn/result/test_result.json"))
    ap.add_argument("--ann", default=os.environ.get(
        "PRIOREDIT_VAL_ANN", f"{RG}/data/swcmp/val_prior_ann.json"))
    ap.add_argument("--out", default=os.path.join(paths.CONFIGS, "confidence_thresholds.json"))
    a = ap.parse_args()

    conf = load_json(a.conf, "val confidence")
    draft = load_json(a.draft, "val sn draft")
    ann = {r["id"]: r for r in load_json(a.ann, "val annotation")["test"]}
    ids = [s for s in conf if s in ann and s in draft]
    if not ids:
        sys.exit("no overlapping study ids across conf / draft / annotation")

    dev = torch.device("cuda:0")
    chx = CheXbert(ckpt_dir="./hf", bert_path="./hf/bert-base-uncased",
                   checkpoint_path="./hf/chexbert.pth", device=dev).to(dev)
    lab = chexbert_present(chx, [ann[s]["finding"] for s in ids])  # current-GT presence truth y
    y_of = {s: {COND2NAME[j]: lab[i, j] for j in range(13)} for i, s in enumerate(ids)}

    # cache prior/draft states once per study
    prior_st = {s: extract(ann[s].get("last_finding", "")) for s in ids}
    draft_st = {s: extract(first(draft[s])) for s in ids}

    print(f"validation calibration, {len(ids)} studies; scope = prior present AND draft silent\n")
    print(f"{'finding':16s}{'n':>5}{'base':>7}{'a_f':>8}{'b_f':>8}{'tau_pos':>9}{'tau_neg':>9}  open")
    cfg = {"_meta": {
        "scope": "prior present AND draft silent (the Eq. 2 coverage scope)",
        "calib": f"fitted on the sw validation split, {len(ids)} studies",
        "pos_rule": "tau_pos = confidence at P(pos|c) = base_in_scope; looser than P=0.5, so more "
                    "persistent findings are covered and tagged 'unchanged'",
        "neg_rule": "tau_neg = confidence at P(pos|c) = base_in_scope/2; the resolved (negative) branch",
        "fields": "n = in-scope validation pairs; base_in_scope = positive rate in scope; "
                  "open_pos/open_neg = whether that branch is enabled; _alt_tau_pos_P50 = the "
                  "alternative operating point at P=0.5, reported in App. F, not adopted",
        "closed": f"findings with < {MIN_PAIRS} clean in-scope pairs are omitted (pleural effusion, "
                  "consolidation): too few validation pairs (App. C)",
        "paper": "Table 6, App. C",
    }}
    for f in COND2NAME:
        xs, ys = [], []
        for s in ids:
            if prior_st[s][f][0] == "pos" and draft_st[s][f][0] == "none":
                xs.append(conf[s][f])
                ys.append(y_of[s][f])
        n = len(xs)
        if n < MIN_PAIRS:
            continue
        x, yv = np.array(xs), np.array(ys)
        base = float(yv.mean())
        a_c, b_c = fit_logistic(x, yv)
        op, on = f in OPEN_POS, f in OPEN_NEG
        row = {"n": n, "base_in_scope": round(base, 3), "open_pos": op, "open_neg": on}
        tpos = tneg = None
        if op:
            tpos = round(float(tau_at(a_c, b_c, base)), 3)
            row["tau_pos"] = tpos
            row["_alt_tau_pos_P50"] = round(float(tau_at(a_c, b_c, 0.5)), 3)
        if on:
            tneg = round(float(tau_at(a_c, b_c, base / 2)), 3)
            row["tau_neg"] = tneg
        cfg[f] = row
        flags = ("pos" if op else "") + ("/neg" if on else "")
        print(f"{f:16s}{n:>5}{base:>7.3f}{a_c:>8.3f}{b_c:>8.2f}"
              f"{(f'{tpos:.3f}' if tpos is not None else '--'):>9}"
              f"{(f'{tneg:.3f}' if tneg is not None else '--'):>9}  {flags or 'closed'}")

    json.dump(cfg, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\nthresholds -> {a.out}")


if __name__ == "__main__":
    main()
