#!/usr/bin/env python
"""Convert an LLM-RG4-flag test output dir (result/test_result.json = hypo, test_refs.json = ref) into
pred.csv/gt.csv ('Report Impression' column) and run the shared RadEval. Worktree-local paths.

  python eval_stage2_radeval.py --infer_dir <savepath> --name <eval_name> [--run]
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import os, json, argparse, subprocess

RADEVAL = os.path.join(paths.SCRIPTS, "radeval_score.py")
METRICEVAL_PY = paths.PY_METRIC
EVAL_ROOT = paths.LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_dir", required=True, help="savepath containing result/test_result.json")
    ap.add_argument("--name", required=True, help="eval dir name")
    ap.add_argument("--run", action="store_true", help="also invoke RadEval (env metriceval)")
    args = ap.parse_args()
    import pandas as pd
    rf = os.path.join(args.infer_dir, "result")
    hypo = json.load(open(f"{rf}/test_result.json"))
    ref_path = f"{rf}/test_refs.json" if os.path.exists(f"{rf}/test_refs.json") else f"{rf}/refs.json"
    ref = json.load(open(ref_path))
    ids = list(hypo.keys())
    preds = [hypo[i][0] if isinstance(hypo[i], list) else hypo[i] for i in ids]
    gts = [ref[i][0] if isinstance(ref[i], list) else ref[i] for i in ids]
    ed = os.path.join(EVAL_ROOT, args.name)
    os.makedirs(ed, exist_ok=True)
    pd.DataFrame({"Report Impression": preds}).to_csv(f"{ed}/pred.csv", index=False)
    pd.DataFrame({"Report Impression": gts}).to_csv(f"{ed}/gt.csv", index=False)
    print(f"wrote {ed}/pred.csv & gt.csv  (n={len(ids)})")
    if args.run:
        subprocess.run([METRICEVAL_PY, RADEVAL, ed], cwd="/tmp",
                       env=dict(os.environ, PYTHONSAFEPATH="1"), check=True)
    else:
        print(f"next (env metriceval): PYTHONSAFEPATH=1 {METRICEVAL_PY} {RADEVAL} {ed}")


if __name__ == "__main__":
    main()
