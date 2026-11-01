#!/usr/bin/env python
"""Predict MS-CXR-T temporal direction with the trained CXR-BERT verifier, then score macro-accuracy.

CURRENT-context source (--current):
  gold            : gold current report (CEILING)
  <drafts.json>   : image-only greedy draft (DEPLOYMENT)
The PRIOR context always comes from the prior report text (never the prior image).

Resampling fusion (--samples samples.json {id:[N drafts]}, --fuse):
  union   : concatenate finding sentences from greedy+samples into one current_ctx (1 verifier call).
            OFF-DISTRIBUTION for the verifier (it was trained on single coherent descriptions) -> weak.
  vote    : run the verifier on EACH source that mentions the finding (each is a coherent in-distribution
            current_ctx), majority-vote the direction. presence handled by "does any source mention it".
            self-consistency at the output level -- the principled default.
  median  : denoise severity -- take the median severity level across mentioning sources, build one clean
            "{sev} {finding}" current_ctx, 1 call. Attacks the "image-only severity is noisy" bottleneck.

Run (see docs/ENVIRONMENTS.md):
  python predict_direction_mscxrt.py --current <drafts.json> --samples <samples.json> --fuse vote --name X
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import json, argparse, os, sys, subprocess, collections
import torch

from build_direction_trainset import FIND_KW, ctx, FINDINGS
from train_direction_verifier import Clf, fmt, LABELS, CXR
from state_extractor import sev_in
from transformers import AutoTokenizer

MANIFEST = os.path.join(paths.OUT, "mscxrt/manifest.jsonl")
OUT = os.path.join(paths.OUT, "models/direction_verifier")
OUTDIR = os.path.join(paths.OUT, "mscxrt")
SCORER = os.path.join(paths.SCRIPTS, "score_mscxrt.py")
PY = paths.PY_LABEL
SEV_WORD = {1: "minimal", 2: "small", 3: "moderate", 4: "large"}


def median_sev(sources, kws):
    """median severity level (1-4) across sources that mention the finding; None if none carry a level."""
    levels = []
    for s in sources:
        cs = ctx(s, kws)
        for sent in cs:
            lv = sev_in(sent)
            if lv:
                levels.append(lv)
    if not levels:
        return None
    levels.sort()
    return levels[len(levels) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default="gold")
    ap.add_argument("--name", default="verif")
    ap.add_argument("--ckpt", default=f"{OUT}/best.pt")
    ap.add_argument("--samples", help="samples.json {id:[N drafts]}")
    ap.add_argument("--fuse", choices=["union", "vote", "median"], default="union")
    a = ap.parse_args()
    samples = json.load(open(a.samples)) if a.samples else None
    dev = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(CXR, trust_remote_code=True)
    model = Clf().to(dev); model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
    drafts = None if a.current == "gold" else json.load(open(a.current))
    manifest = [json.loads(l) for l in open(MANIFEST)]

    # 1) build items: (pair_id, fd, current_ctx). vote may emit several items per (pair,fd).
    items = []
    for r in manifest:
        prior = r["prior_report"]
        dic = r["current_dicom"].split("/")[-1]
        if drafts is None:
            greedy = r["current_report"]
        else:
            greedy = drafts.get(dic) or drafts.get(str(r["current_study_id"])) or drafts.get(r["pair_id"]) or ""
            if isinstance(greedy, list):
                greedy = greedy[0]
        samp = []
        if samples is not None:
            st = samples.get(dic) or samples.get(str(r["current_study_id"])) or []
            samp = st if isinstance(st, list) else [st]
        sources = [greedy] + samp
        for fd in r["labels"]:
            pc = " . ".join(ctx(prior, FIND_KW[fd]))
            if a.fuse == "vote" and samples is not None:
                ccs = []
                for s in sources:
                    cs = ctx(s, FIND_KW[fd])
                    if cs:
                        ccs.append(" . ".join(cs))
                if not ccs:
                    ccs = [""]                         # nobody mentions it -> prior-only single vote
                for cc in ccs:
                    items.append((r["pair_id"], fd, pc, cc))
            elif a.fuse == "median" and samples is not None:
                lv = median_sev(sources, FIND_KW[fd])
                present = any(ctx(s, FIND_KW[fd]) for s in sources)
                cc = (f"{SEV_WORD.get(lv, '')} {fd.replace('_', ' ')}".strip()) if present else ""
                items.append((r["pair_id"], fd, pc, cc))
            else:  # union
                seen, cc_sents = set(), []
                for s in sources:
                    for sent in ctx(s, FIND_KW[fd]):
                        if sent not in seen:
                            seen.add(sent); cc_sents.append(sent)
                items.append((r["pair_id"], fd, pc, " . ".join(cc_sents)))

    # 2) batched verifier over items
    preds = []
    for i in range(0, len(items), 128):
        chunk = items[i:i + 128]
        txt = [fmt({"finding": fd, "prior_ctx": pc, "current_ctx": cc}) for _, fd, pc, cc in chunk]
        enc = tok([t[0] for t in txt], [t[1] for t in txt], truncation=True, max_length=192,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            preds += model(**enc).argmax(1).cpu().tolist()

    # 3) aggregate per (pair,fd)
    votes = collections.defaultdict(list)
    for (pid, fd, _, _), yi in zip(items, preds):
        votes[(pid, fd)].append(yi)
    pred = {}
    for (pid, fd), ys in votes.items():
        c = collections.Counter(ys)
        top = max(c.values())
        winners = [k for k, v in c.items() if v == top]
        yi = min(winners, key=lambda k: 0 if LABELS[k] == "stable" else 1) if len(winners) > 1 else winners[0]
        pred.setdefault(pid, {})[fd] = LABELS[yi]

    pp = f"{OUTDIR}/pred_{a.name}.json"
    json.dump(pred, open(pp, "w"))
    print(f"### direction verifier (current={a.current}, fuse={a.fuse}) -> {pp} ###")
    subprocess.run([PY, SCORER, "--pred", pp])


if __name__ == "__main__":
    main()
