#!/usr/bin/env python
"""Temporal hallucination rate: a temporal claim is HALLUCINATED if the PRIOR report gives no basis
for it (direction-aware):
  - unchanged/increased/decreased/resolved on finding X  -> requires X PRESENT in prior (else made up)
  - new/interval-development on X                         -> requires X ABSENT  in prior (else not new)

Prior presence = extract() on the prior report (last_finding). Measures how often a report asserts a
comparison the prior can't support. A grounded decoupled editor -> ~0; end-to-end prior-conditioned
generation (sw) hallucinates temporal because it does not check the prior.

Run: python temporal_halluc_eval.py --pred <report json> --ann <sw annotation> --name <label>
"""
import os as _os, sys as _sys
_scripts = _os.path.dirname(_os.path.abspath(__file__))
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402,F401  (also puts scripts/* groups on sys.path)
import json, re, argparse, os, sys, collections
from state_extractor import extract, FIND_KW

# UNAMBIGUOUS comparison words only (exclude descriptive enlarged/larger/smaller/greater/bigger which
# describe current state, not a change) -> avoids counting descriptive language as a temporal claim.
PRESENT_REQ = re.compile(r"\b(unchanged|stable|stably|persistent|persistence|persisting|"
                         r"unaltered|increased|increasing|decreased|decreasing|reduced|worsen\w*|"
                         r"improv\w*|resolved|resolution|resolving|redemonstrat\w*|reaccumulat\w*|"
                         r"progression|progressive|previously)\b")
ABSENT_REQ = re.compile(r"\b(new|newly|interval development|developed|development of|not previously)\b")
FINDINGS = [f for f in FIND_KW if f not in ("support_devices",)]


def prior_present(prior_text):
    e = extract(prior_text)
    return {f: (e[f][0] == "pos") for f in FINDINGS}


def claims_with_type(text):
    """per finding -> set of {'present_req','absent_req'} claim types found in a sentence mentioning it."""
    out = collections.defaultdict(set)
    for s in re.split(r"[.;\n]", (text or "").lower()):
        pr = bool(PRESENT_REQ.search(s)); ab = bool(ABSENT_REQ.search(s))
        if not (pr or ab):
            continue
        for fd in FINDINGS:
            if any(re.search(k, s) for k in FIND_KW[fd]):
                if pr:
                    out[fd].add("present_req")
                if ab:
                    out[fd].add("absent_req")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--ann", required=True)
    ap.add_argument("--name", default="pred")
    a = ap.parse_args()
    pred = json.load(open(a.pred))
    priors = {}
    for r in json.load(open(a.ann))["test"]:
        lf = r.get("last_finding", "")
        priors[str(r["id"])] = "" if lf.strip() == "s" else lf
        priors[str(r["study_id"])] = priors[str(r["id"])]
    total = halluc = 0
    by_type = collections.Counter()
    for sid, rep in pred.items():
        rep = rep[0] if isinstance(rep, list) else rep
        pt = priors.get(str(sid))
        if pt is None or not pt.strip():
            continue                         # no prior text -> skip (can't judge)
        pp = prior_present(pt)
        for fd, types in claims_with_type(rep).items():
            for t in types:
                total += 1
                if t == "present_req" and not pp[fd]:
                    halluc += 1; by_type["present_req_but_prior_absent"] += 1
                elif t == "absent_req" and pp[fd]:
                    halluc += 1; by_type["new_but_prior_present"] += 1
    rate = halluc / total * 100 if total else 0.0
    print(f"=== temporal hallucination | {a.name} ===")
    print(f"  temporal claims checked: {total}")
    print(f"  hallucinated (prior can't support): {halluc} ({rate:.1f}%)")
    print(f"  breakdown: {dict(by_type)}")


if __name__ == "__main__":
    main()
