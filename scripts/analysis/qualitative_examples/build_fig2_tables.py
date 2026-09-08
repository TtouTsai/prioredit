#!/usr/bin/env python
"""Build the Figure-2 qualitative candidate tables (4 CSVs, shared schema).

  table1_positive.csv    true=resolved & sw=persistent & PriorEdit=resolved   (strict; n=6)
  table1n_new.csv        true=new & PriorEdit=new & sw asserted a different class
  table1u_unchanged.csv  true=persistent & PriorEdit=persistent & sw asserted a different class
  table3_tau_neg_fp.csv  true=persistent & PriorEdit resolved via tau_neg     (n=5)
  table4_tau_pos_fp.csv  true=resolved & PriorEdit recovered pos via tau_pos  (primary n=129)
  table4b_tau_pos_fp_gtabsent.csv   same but GT-absent (none/neg/unc) instead of true=resolved

Table 2 (multi-finding studies) is intentionally not produced: no study has >=2 findings meeting
the table-1 condition.

All sentences are VERBATIM (see sentences.py). Missing values are the literal string 'N/A';
nothing is imputed. Reads only; writes new CSVs into --outdir.

  python build_fig2_tables.py --outdir <dir>
"""
import os as _os, sys as _sys
_scripts = _os.path.dirname(_os.path.abspath(__file__))
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402,F401  (also puts scripts/* groups on sys.path)
import os, sys, csv, json, argparse, statistics, collections

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from editor_paths import load_all, editor_paths                      # noqa: E402
from sentences import joined, wordcount                              # noqa: E402
from sw_artifact import classify as sw_classify                      # noqa: E402
from temporal_accuracy import asserted_transitions, true_transitions, FIND  # noqa: E402
from rule_editor import EDIT_FINDINGS                                  # noqa: E402

NA = "N/A"

COLS = [
    "study_id", "dicom_id", "finding",
    "prior_sentence", "gt_sentence", "sw_sentence", "prioredit_sentence",
    "c_f", "tau_pos", "tau_neg",
    "prioredit_path",
    "true_transition", "sw_asserted", "prioredit_asserted",
    "sw_label_verdict", "sw_trigger_sentence", "sw_verdict_reason",
    "prior_words", "gt_words", "sw_words", "prioredit_words",
    "contrast_words",           # sw + prioredit words, the table-1 sort key
]


def fmt(v, nd=3):
    return NA if v is None else f"{v:.{nd}f}"


def build_rows(D, keep):
    """keep(finding, true, sw_asserted, ed_asserted, path, gt_label) -> bool"""
    ids = [i for i in D["prior_lab"] if i in D["gt_lab"] and i in D["editor"] and i in D["sw"]]
    rows = []
    for sid in sorted(ids):
        meta = D["meta"][sid]
        lf = meta.get("last_finding", "")
        prior_text = "" if lf.strip() == "s" else lf
        gt_text = meta.get("finding", "")
        conf = D["conf"].get(sid) or {}
        paths = editor_paths(D["sn"][sid], prior_text, conf, D["thresholds"])
        tt = true_transitions(D["prior_lab"][sid], D["gt_lab"][sid])
        ae = asserted_transitions(D["editor"][sid])
        aw = asserted_transitions(D["sw"][sid])
        for fd in EDIT_FINDINGS:
            if fd not in FIND:
                continue
            path, _state, _tau = paths[fd]
            if not keep(fd, tt[fd], aw[fd], ae[fd], path, D["gt_lab"][sid].get(fd)):
                continue
            th = D["thresholds"].get(fd, {})
            tau_pos = th.get("tau_pos") if th.get("open_pos") else None
            tau_neg = th.get("tau_neg") if th.get("open_neg") else None
            s_prior = joined(prior_text, fd) if prior_text else "no prior"
            s_gt = joined(gt_text, fd)
            s_sw = joined(D["sw"][sid], fd)
            s_ed = joined(D["editor"][sid], fd)
            v_verdict, v_trig, v_reason = (
                sw_classify(D["sw"][sid], fd, aw[fd]) if aw[fd] != "none" else
                ("n/a", "N/A", "sw asserted nothing for this finding"))
            rows.append({
                "study_id": meta.get("study_id", NA),
                "dicom_id": sid,
                "finding": fd,
                "prior_sentence": s_prior,
                "gt_sentence": s_gt,
                "sw_sentence": s_sw,
                "prioredit_sentence": s_ed,
                "c_f": fmt(conf.get(fd)),
                "tau_pos": fmt(tau_pos),
                "tau_neg": fmt(tau_neg),
                "prioredit_path": path,
                "true_transition": tt[fd],
                "sw_asserted": aw[fd],
                "prioredit_asserted": ae[fd],
                "sw_label_verdict": v_verdict,
                "sw_trigger_sentence": v_trig,
                "sw_verdict_reason": v_reason,
                "prior_words": wordcount(s_prior),
                "gt_words": wordcount(s_gt),
                "sw_words": wordcount(s_sw),
                "prioredit_words": wordcount(s_ed),
                "contrast_words": wordcount(s_sw) + wordcount(s_ed),
            })
    return rows


def write(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {len(rows):>4} rows -> {path}")


def margin_stats(rows, which):
    """|c_f - tau| distribution for the threshold the row actually crossed."""
    d = []
    for r in rows:
        c = r["c_f"]
        t = r[which]
        if c == NA or t == NA:
            continue
        d.append(float(c) - float(t))
    if not d:
        return None
    a = [abs(x) for x in d]
    buckets = collections.OrderedDict()
    for lo, hi in [(0, .5), (.5, 1), (1, 2), (2, 4), (4, 1e9)]:
        buckets[f"{lo}-{'inf' if hi > 1e8 else hi}"] = sum(1 for x in a if lo <= x < hi)
    return {"n": len(d), "min": min(a), "median": statistics.median(a), "max": max(a),
            "mean": statistics.mean(a), "buckets": buckets,
            "signed_min": min(d), "signed_max": max(d)}


def show(name, st):
    if not st:
        print(f"\n{name}: no margin data")
        return
    print(f"\n{name} (n={st['n']}) |c_f - tau|: "
          f"min={st['min']:.3f} median={st['median']:.3f} mean={st['mean']:.3f} max={st['max']:.3f}")
    for k, v in st["buckets"].items():
        bar = "#" * int(40 * v / max(st["n"], 1))
        print(f"    {k:>8} : {v:>4}  {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(_HERE, "out"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    D = load_all()
    print(f"loaded; eligible-with-prior = {len(D['prior_lab'])}")

    # ---- Table 1: strict positive contrast ----
    t1 = build_rows(D, lambda fd, t, sw, ed, p, g:
                    t == "resolved" and sw == "persistent" and ed == "resolved")
    t1.sort(key=lambda r: (r["contrast_words"], r["sw_words"] + r["prioredit_words"]))
    write(t1, os.path.join(a.outdir, "table1_positive.csv"))

    # ---- Table 1n / 1u: same strict contrast for the other two temporal classes ----
    # (asked for after the first pass: table 1 was resolved-only). "sw wrong in a DIRECTED way"
    # = sw asserted some other change class, not merely silence.
    t1n = build_rows(D, lambda fd, t, sw, ed, p, g:
                     t == "new" and ed == "new" and sw not in ("new", "none"))
    t1n.sort(key=lambda r: (r["contrast_words"], r["sw_words"] + r["prioredit_words"]))
    write(t1n, os.path.join(a.outdir, "table1n_new.csv"))

    t1u = build_rows(D, lambda fd, t, sw, ed, p, g:
                     t == "persistent" and ed == "persistent" and sw not in ("persistent", "none"))
    t1u.sort(key=lambda r: (r["contrast_words"], r["sw_words"] + r["prioredit_words"]))
    # sw's label here is often an artifact of `appear\w*` living in the NEW regex ("appears
    # unchanged" -> scored new). Those are NOT sw errors and must not be shown as such.
    t1u_ok = [r for r in t1u if r["sw_label_verdict"] == "genuine"]
    t1u_weak = [r for r in t1u if r["sw_label_verdict"] == "weak"]
    t1u_bad = [r for r in t1u if r["sw_label_verdict"] == "artifact"]
    write(t1u_ok, os.path.join(a.outdir, "table1u_unchanged.csv"))
    write(t1u_weak, os.path.join(a.outdir, "table1u_unchanged_WEAK.csv"))
    write(t1u_bad, os.path.join(a.outdir, "table1u_unchanged_EXCLUDED_artifact.csv"))

    # ---- Table 3: tau_neg false resolve ----
    t3 = build_rows(D, lambda fd, t, sw, ed, p, g:
                    t == "persistent" and ed == "resolved" and p == "tau_neg_resolve")
    t3.sort(key=lambda r: abs(float(r["c_f"]) - float(r["tau_neg"])))
    write(t3, os.path.join(a.outdir, "table3_tau_neg_fp.csv"))

    # ---- Table 4 (primary, n=129): true=resolved, recovered via tau_pos ----
    t4 = build_rows(D, lambda fd, t, sw, ed, p, g:
                    t == "resolved" and ed == "persistent" and p == "tau_pos_recover")
    t4.sort(key=lambda r: abs(float(r["c_f"]) - float(r["tau_pos"])))
    write(t4, os.path.join(a.outdir, "table4_tau_pos_fp.csv"))

    # ---- Table 4b (secondary): GT-absent instead of true=resolved ----
    t4b = build_rows(D, lambda fd, t, sw, ed, p, g:
                     p == "tau_pos_recover" and g != "pos")
    t4b.sort(key=lambda r: abs(float(r["c_f"]) - float(r["tau_pos"])))
    write(t4b, os.path.join(a.outdir, "table4b_tau_pos_fp_gtabsent.csv"))

    print("\n" + "=" * 68)
    print("MARGIN DISTRIBUTIONS (how far the wrong calls sit from the threshold)")
    print("=" * 68)
    show("Table 3  tau_neg mis-resolve", margin_stats(t3, "tau_neg"))
    show("Table 4  tau_pos mis-recover (primary, true=resolved)", margin_stats(t4, "tau_pos"))
    show("Table 4b tau_pos mis-recover (GT-absent)", margin_stats(t4b, "tau_pos"))

    # control: tau_pos recoveries that were CORRECT, to show the margin does not separate them
    ctrl = build_rows(D, lambda fd, t, sw, ed, p, g:
                      p == "tau_pos_recover" and t == "persistent")
    show("CONTROL  tau_pos recover that was CORRECT (true=persistent)", margin_stats(ctrl, "tau_pos"))

    print("\nper-finding counts:")
    for nm, rr in [("table1", t1), ("table1n", t1n), ("table1u", t1u_ok), ("table1u_weak", t1u_weak), ("table1u_excl", t1u_bad),
                   ("table3", t3), ("table4", t4), ("table4b", t4b)]:
        print(f"  {nm:8s} n={len(rr):>4}  {dict(collections.Counter(r['finding'] for r in rr))}")
    print("\nNOTE: table2 (>=2 qualifying findings in one study) is empty -> not written.")


if __name__ == "__main__":
    main()
