#!/usr/bin/env python
"""Temporal hallucination rate with the OFFICIAL Stanford CheXpert labeler (dockerized) as the prior GT.

Most authoritative version: prior-presence comes from the official rule labeler (mention phrase lists +
NegBio negation/uncertainty), independent of both our regex extractor AND the neural CheXbert. A temporal
claim is hallucinated if the prior labels don't support it:
  present_req (unchanged/inc/dec/resolved) on X -> requires official prior label[X] present
  absent_req  (new)                        on X -> requires official prior label[X] not-present
Uncertain (-1) is treated as present for present_req (benefit of the doubt, matches CheXbert binarization).

Prior labels come from chexpert_label.py output {id: {finding: pos|neg|unc|none}}.

Run (env rrg_dpo_new):
  python temporal_halluc_official.py --pred <report json> --ann <sw ann> --prior_labels <labels json> --name X
"""
import os as _os, sys as _sys
_scripts = _os.path.dirname(_os.path.abspath(__file__))
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402,F401  (also puts scripts/* groups on sys.path)
import json, argparse, os, sys, collections

from temporal_halluc_eval import claims_with_type
from chexpert_label import CAT2FD  # finding keys used by the labeler output

FINDINGS = set(CAT2FD.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--ann", required=True)
    ap.add_argument("--prior_labels", required=True, help="chexpert_label.py output {id:{finding:state}}")
    ap.add_argument("--name", default="pred")
    ap.add_argument("--present_includes_uncertain", action="store_true", default=True)
    a = ap.parse_args()
    pred = json.load(open(a.pred))
    labels = json.load(open(a.prior_labels))
    # map both id and study_id to the labeled prior (labels are keyed by the annotation 'id')
    ann = json.load(open(a.ann))["test"]
    id2sid = {str(r["id"]): str(r["study_id"]) for r in ann}
    # prior present per finding: official pos (or uncertain) counts as present
    def prior_present(key):
        lab = labels.get(str(key))
        if lab is None:
            return None
        return {f: (lab.get(f) in ("pos", "unc")) for f in FINDINGS}

    total = halluc = 0
    by_type = collections.Counter(); byfd = collections.Counter(); byfd_tot = collections.Counter()
    for sid, rep in pred.items():
        rep = rep[0] if isinstance(rep, list) else rep
        pp = prior_present(sid)
        if pp is None:                       # pred keyed by dicom id == labels key
            pp = prior_present(id2sid.get(str(sid), ""))
        if pp is None:
            continue
        for fd, types in claims_with_type(rep).items():
            if fd not in FINDINGS:
                continue
            present = pp[fd]
            for t in types:
                total += 1; byfd_tot[fd] += 1
                bad = (t == "present_req" and not present) or (t == "absent_req" and present)
                if bad:
                    halluc += 1; byfd[fd] += 1
                    by_type[("present_req_but_prior_absent" if t == "present_req"
                             else "new_but_prior_present")] += 1
    rate = halluc / total * 100 if total else 0.0
    print(f"=== OFFICIAL CheXpert-labeler prior GT | temporal hallucination | {a.name} ===")
    print(f"  temporal claims checked: {total}")
    print(f"  hallucinated (official prior can't support): {halluc} ({rate:.1f}%)")
    print(f"  breakdown: {dict(by_type)}")
    print(f"  by finding (halluc/total):")
    for fd, _ in byfd_tot.most_common():
        print(f"    {fd:22s} {byfd[fd]:3d}/{byfd_tot[fd]:3d}")


if __name__ == "__main__":
    main()
