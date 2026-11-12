#!/usr/bin/env python
"""Score the LLM-parser asserted transitions with the UNCHANGED TCA scorer (paper App. E).

temporal_assessment.score() calls asserted_transitions(report) internally. To feed it the cached LLM
transitions WITHOUT editing that module, we temporarily rebind TA.asserted_transitions to an identity
(the cache already holds per-finding transitions), then restore it; the true-transition side and all
counting logic stay byte-identical to the paper's scorer.

The LLM caches (data/llm_parser/asserted_*_<tag>.json) are shipped and text-free, so the LLM-parser
columns and both gaps reproduce from this release alone. Only the regex GT-self cell needs the GT report
text (the LLM-RG4 annotation, MIMIC-CXR under a DUA); without $PRIOREDIT_LLMRG4 it is printed as n/a.

  python llm_parser_eval.py --tag qwen3        # or: --tag medgemma
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402
import os, sys, json, argparse
import temporal_assessment as TA  # UNCHANGED shared module
sys.path.insert(0, _here)
from llm_parser_extract import load_inputs, _gen, RG, OUTDIR  # noqa: E402


def restrict(d, ids):
    return {i: d[i] for i in ids}


def llm_rows(cache, ids, prior, gt):
    orig = TA.asserted_transitions
    try:
        TA.asserted_transitions = lambda rep: rep  # cache already holds the transitions
        rows, _ = TA.score(restrict(cache, ids), restrict(prior, ids), restrict(gt, ids))
    finally:
        TA.asserted_transitions = orig
    return rows


def regex_rows(texts, ids, prior, gt):
    return TA.score(restrict(texts, ids), restrict(prior, ids), restrict(gt, ids))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen3", help="cache tag, e.g. qwen3 or medgemma")
    a = ap.parse_args()
    prior = json.load(open(os.path.join(paths.LABELS, "prior_chexpert_labels.json")))
    gt = json.load(open(os.path.join(paths.LABELS, "gt_chexpert_labels.json")))
    if os.path.exists(os.path.join(RG, "data/swcmp/annotation.json")):
        _, texts = load_inputs()
    else:  # release-only: PriorEdit/sw text is shipped, GT text is not
        cohort = [i for i in prior if i in gt]
        texts = {"gtself": None,
                 "prioredit": _gen(os.path.join(paths.GENERATIONS, "prioredit.json"), cohort),
                 "sw": _gen(os.path.join(paths.GENERATIONS, "llmrg4_sw.json"), cohort)}

    caches = {}
    for s in ("gtself", "prioredit", "sw"):
        p = os.path.join(OUTDIR, f"asserted_{s}_{a.tag}.json")
        if not os.path.exists(p):
            sys.exit(f"missing cache: {p}  (produce with llm_parser_extract.py --tag {a.tag})")
        caches[s] = json.load(open(p))
    ids = set(caches["gtself"]) & set(prior) & set(gt) & set(texts["prioredit"]) & set(texts["sw"])
    if texts["gtself"] is not None:
        ids &= set(texts["gtself"])
    ids = sorted(ids)
    print(f"tag={a.tag}: scored on the same {len(ids)} studies for every system/parser\n")

    def chg(system, parser):
        rows = (regex_rows(texts[system], ids, prior, gt) if parser == "regex"
                else llm_rows(caches[system], ids, prior, gt))
        return rows["change(micro)"][0] * 100

    print(f"{'':10s}{'regex':>9}{'LLM':>9}")
    gt_l = chg("gtself", "LLM")
    gt_r = f"{chg('gtself', 'regex'):>8.1f}%" if texts["gtself"] is not None else f"{'n/a':>9}"
    print(f"{'GT self':10s}{gt_r}{gt_l:>8.1f}%   (App. E: not a regex artifact)")
    for s in ("prioredit", "sw"):
        print(f"{s:10s}{chg(s,'regex'):>8.1f}%{chg(s,'LLM'):>8.1f}%")
    for parser in ("regex", "LLM"):
        gap = chg("prioredit", parser) - chg("sw", parser)
        print(f"  PriorEdit-sw gap ({parser}): {gap:+.1f}")


if __name__ == "__main__":
    main()
