#!/usr/bin/env python
"""RaTE column of Table 7 (paper App. F): the image-only bases Qwen3-VL-2B and LLM-RG4 sn on the
1337 with-prior studies.

RaTEScore (the RaTEScore package) of each generated findings text against the current GT findings,
both with the leading "finding :" marker removed; reported as the mean over studies
(paper: Qwen3-VL-2B .461, sn .597). radeval_score.py on the make_eval_dir.py dirs, which keep the
marker, gives slightly lower values (Qwen3-VL .458, sn .596) with the same ordering.

The other Table 7 columns: presence F1/P/R from paper_chexbert_eval.py on
`make_eval_dir.py --with_prior` dirs, and Change from temporal_assessment.py (#pred of change(micro)).

  python imageonly_bases_rate.py --ann $PRIOREDIT_LLMRG4/data/swcmp/annotation.json

Env: RaTEScore (pip install RaTEScore) with torch; a GPU is used if available.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import os, sys, json, argparse

BASES = [("Qwen3-VL-2B", "qwen3vl_imageonly.json"), ("LLM-RG4 sn", "llmrg4_sn.json")]


def strip(t):
    t = t[0] if isinstance(t, list) else t
    t = (t or "").strip()
    return t[len("finding :"):].strip() if t.lower().startswith("finding :") else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="LLM-RG4 swcmp annotation.json (current GT findings)")
    a = ap.parse_args()
    try:  # RaTEScore's sentencizer logs every sentence at DEBUG level; keep report text off the console
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    except ImportError:
        pass
    from RaTEScore import RaTEScore
    ann = {str(r["id"]): r for r in json.load(open(a.ann))["test"]}
    prior = json.load(open(os.path.join(paths.LABELS, "prior_chexpert_labels.json")))
    gt = json.load(open(os.path.join(paths.LABELS, "gt_chexpert_labels.json")))
    gens = {name: json.load(open(os.path.join(paths.GENERATIONS, fn))) for name, fn in BASES}
    ids = [i for i in gens["Qwen3-VL-2B"] if i in ann and i in prior and i in gt
           and all(i in g for g in gens.values())]
    print(f"n={len(ids)} with-prior studies")
    ref = [strip(ann[i].get("finding", "")) for i in ids]
    rate = RaTEScore()
    for name, g in gens.items():
        sc = [float(x) for x in rate.compute_score([strip(g[i]) for i in ids], ref)]
        print(f"  {name:12s} RaTE = {sum(sc) / len(sc):.4f}")


if __name__ == "__main__":
    main()
