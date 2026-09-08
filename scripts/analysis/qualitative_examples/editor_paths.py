#!/usr/bin/env python
"""Reconstruct, per (study, finding), which branch the rule editor took.

rule_editor.py logs nothing, so we re-derive the branch by mirroring confidence_state()'s
decision tree exactly (it is deterministic). Validated by regenerating the adopted
combo5_bal.json byte-for-byte with the same flags (1698/1698 identical).

ADOPTED FLAGS (verified, NOT --official despite swcmp_findings.md:429):
  --confidence --thresholds --inplace --high_conf --keep_unchanged --sev_jump 2

Paths:
  draft_pos        draft states the finding positive  -> authoritative, no confidence used
  draft_neg        draft explicitly negates it        -> authoritative (source of most 'resolved')
  tau_pos_recover  draft silent, prior=pos, c_f > tau_pos -> recovered to pos ('unchanged')
  tau_neg_resolve  draft silent, prior=pos, c_f < tau_neg -> asserted absent ('now resolved')
  abstain_in_scope draft silent, prior=pos, in scope but no threshold crossed / gate closed
  out_of_scope     draft silent and prior not pos -> confidence never consulted
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import os, sys, json


from rule_editor import EDIT_FINDINGS, has_prior          # noqa: E402
from state_extractor import extract                          # noqa: E402

RG = paths.LLMRG4

P_ANN = f"{RG}/data/swcmp/annotation.json"
P_SN = os.path.join(paths.GENERATIONS, "llmrg4_sn.json")
P_SW = os.path.join(paths.GENERATIONS, "llmrg4_sw.json")
P_EDITOR = os.path.join(paths.GENERATIONS, "prioredit.json")
P_CONF = os.path.join(paths.ROOT, "data", "finding_confidence.json")
P_THRESH = os.path.join(paths.CONFIGS, "confidence_thresholds.json")
P_PRIOR_LAB = os.path.join(paths.LABELS, "prior_chexpert_labels.json")
P_GT_LAB = os.path.join(paths.LABELS, "gt_chexpert_labels.json")


def as_text(v):
    return v[0] if isinstance(v, list) else v


def load_all():
    ann = json.load(open(P_ANN))["test"]
    meta = {r["id"]: r for r in ann}
    return {
        "meta": meta,
        "sn": {k: as_text(v) for k, v in json.load(open(P_SN)).items()},
        "sw": {k: as_text(v) for k, v in json.load(open(P_SW)).items()},
        "editor": {k: as_text(v) for k, v in json.load(open(P_EDITOR)).items()},
        "conf": json.load(open(P_CONF)),
        "thresholds": json.load(open(P_THRESH)),
        "prior_lab": json.load(open(P_PRIOR_LAB)),
        "gt_lab": json.load(open(P_GT_LAB)),
    }


def editor_paths(draft, prior_text, conf, thresholds):
    """Mirror of rule_editor.confidence_state(), returning (path, state, tau_used) per finding.

    Mirrors the real gate ordering: open_pos is checked before open_neg, and a finding whose
    prior is not 'pos' never consults c_f at all."""
    g = extract(draft)
    p = extract(prior_text)
    conf = conf or {}
    out = {}
    prior_ok = has_prior(prior_text)
    for fd in EDIT_FINDINGS:
        gs = g[fd][0]
        if gs == "pos":
            out[fd] = ("draft_pos", "pos", None)
            continue
        if gs == "neg":
            out[fd] = ("draft_neg", "neg", None)
            continue
        th = thresholds.get(fd, {})
        c = conf.get(fd)
        if p[fd][0] == "pos" and c is not None:
            if th.get("open_pos") and th.get("tau_pos") is not None and c > th["tau_pos"]:
                out[fd] = ("tau_pos_recover", "pos", th["tau_pos"])
            elif th.get("open_neg") and th.get("tau_neg") is not None and c < th["tau_neg"]:
                out[fd] = ("tau_neg_resolve", "neg", th["tau_neg"])
            else:
                out[fd] = ("abstain_in_scope", "none", None)
        else:
            out[fd] = ("out_of_scope", "none", None)
    if not prior_ok:
        # editor returns the draft untouched -> no coverage branch could have fired
        out = {fd: (("draft_pos" if v[1] == "pos" and v[0] == "draft_pos" else
                     "draft_neg" if v[0] == "draft_neg" else "no_prior"), v[1], v[2])
               for fd, v in out.items()}
    return out
