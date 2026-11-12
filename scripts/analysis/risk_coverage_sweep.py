#!/usr/bin/env python
"""Risk-coverage sweep for PriorEdit's confidence coverage (paper App. F, threshold operating point).

For a grid of target probabilities P*, set each opened finding's positive threshold to
tau_pos = (logit(P*) - b_f) / a_f, re-run the rule editor, and score change precision vs the number of
asserted comparisons. The per-finding logistic (a_f, b_f) is recovered from the two stored threshold
points in configs/confidence_thresholds.json (tau_pos at P*=base_in_scope and _alt_tau_pos_P50 at
P*=0.5), so no GPU or re-calibration is needed. The point is that admitting more coverage (lower P*)
raises the 'unchanged' volume without moving change-class precision.

Pure CPU. The sn draft, the sw generations and the cached confidence probe are shipped
(data/generations/, data/finding_confidence.json). Only the annotation (prior report text, MIMIC-CXR under
a PhysioNet DUA, not redistributed) comes from the LLM-RG4 checkout. Baselines are single reference points.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots from PRIOREDIT_* env vars; also puts scripts/* groups on sys.path)
import os, sys, json, math, copy, argparse, subprocess
import temporal_assessment as TA  # noqa: E402  (shared metric module)

RG = paths.LLMRG4
RULE_EDITOR = os.path.join(paths.SCRIPTS, "method", "rule_editor.py")


def logit(p):
    return math.log(p / (1 - p))


def recover_logistic(cfg):
    """Per opened finding, recover (a, b) from the two (P*, tau) points where they differ."""
    logi = {}
    for f, d in cfg.items():
        if f == "_meta" or not isinstance(d, dict) or not d.get("open_pos"):
            continue
        t_base, t_50, base = d.get("tau_pos"), d.get("_alt_tau_pos_P50"), d.get("base_in_scope")
        if t_base is None or t_50 is None or base is None or abs(t_base - t_50) < 1e-6:
            continue  # degenerate (e.g. base == 0.5): tau does not move with P*, leave fixed
        a = logit(base) / (t_base - t_50)   # logit(0.5) = 0, so tau_50 = -b/a and tau_base = (logit(base)-b)/a
        b = -a * t_50
        logi[f] = (a, b)
    return logi


def run_point(pstar, cfg, logi, args, out_dir):
    thr = copy.deepcopy(cfg)
    for f, (a, b) in logi.items():
        thr[f]["tau_pos"] = round((logit(pstar) - b) / a, 4)
    tf = os.path.join(out_dir, f"thr_p{int(pstar*100)}.json")
    of = os.path.join(out_dir, f"edit_p{int(pstar*100)}.json")
    json.dump(thr, open(tf, "w"))
    subprocess.run([sys.executable, RULE_EDITOR, "--drafts", args.drafts, "--ann", args.ann,
                    "--confidence", args.conf, "--thresholds", tf, "--inplace", "--high_conf",
                    "--keep_unchanged", "--sev_jump", "2", "--out", of],
                   check=True, capture_output=True)
    rows, _ = TA.score(json.load(open(of)), args._prior, args._gt)
    cp, cr, _, cpred, _, ctp = rows["change(micro)"]
    asserted = sum(rows[c][3] for c in ["new", "resolved", "persistent"])
    return dict(pstar=pstar, change_P=cp, change_R=cr, change_pred=cpred,
                resolved_tp=rows["resolved"][5], asserted=asserted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=os.path.join(paths.LABELS, "prior_chexpert_labels.json"))
    ap.add_argument("--gt", default=os.path.join(paths.LABELS, "gt_chexpert_labels.json"))
    ap.add_argument("--config", default=os.path.join(paths.CONFIGS, "confidence_thresholds.json"))
    ap.add_argument("--drafts", default=os.path.join(paths.GENERATIONS, "llmrg4_sn.json"))
    ap.add_argument("--ann", default=f"{RG}/data/swcmp/annotation.json")
    ap.add_argument("--conf", default=os.path.join(paths.ROOT, "data", "finding_confidence.json"))
    ap.add_argument("--sw", default=os.path.join(paths.GENERATIONS, "llmrg4_sw.json"),
                    help="sw generations, for the reference point")
    ap.add_argument("--grid", default="0.15,0.2,0.281,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--out", default=os.path.join(paths.OUT, "risk_coverage"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a._prior = json.load(open(a.prior))
    a._gt = json.load(open(a.gt))
    cfg = json.load(open(a.config))
    logi = recover_logistic(cfg)
    print("sweepable findings:", list(logi), file=sys.stderr)

    print(f"{'P*':>7}{'change_P':>10}{'change_R':>10}{'#change':>9}{'#resTP':>8}{'#asserted':>11}")
    results = []
    for p in [float(x) for x in a.grid.split(",")]:
        r = run_point(p, cfg, logi, a, a.out)
        results.append(r)
        print(f"{p:>7.3f}{r['change_P']*100:>9.0f}%{r['change_R']*100:>9.0f}%"
              f"{r['change_pred']:>9}{r['resolved_tp']:>8}{r['asserted']:>11}")

    if os.path.exists(a.sw):
        rows, _ = TA.score(json.load(open(a.sw)), a._prior, a._gt)
        asserted = sum(rows[c][3] for c in ["new", "resolved", "persistent"])
        print(f"\nreference  sw: change_P={rows['change(micro)'][0]*100:.0f}%  #asserted={asserted}",
              file=sys.stderr)
    json.dump(results, open(os.path.join(a.out, "risk_coverage_results.json"), "w"), indent=2)
    print(f"\nsaved -> {os.path.join(a.out, 'risk_coverage_results.json')}")


if __name__ == "__main__":
    main()
