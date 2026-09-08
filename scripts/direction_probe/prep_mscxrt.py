#!/usr/bin/env python
"""Build the MS-CXR-T temporal-progression eval manifest for our pipeline.

For each labeled (image-pair, finding) instance: resolve current image path + prior report text
(our pipeline's inputs) + gold progression class. Output one JSONL row per PAIR with the per-finding
gold labels, so downstream: prior report + current image -> our draft -> verifier -> per-finding
{improving/stable/worsening} -> scored by score_mscxrt.py.

Run: python prep_mscxrt.py
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import pandas as pd, os, json

D = os.path.join(paths.DATA, "ms-cxr-t/ms-cxr-t-learning-to-exploit-temporal-structure-for-biomedical-vision-language-processing-1.0.0")
CSV = f"{D}/MS_CXR_T_temporal_image_classification_v1.0.0.csv"
IMG = os.path.join(paths.DATA, "mimic-cxr-jpg/files")
REP = os.path.join(paths.DATA, "mimic-cxr-reports/files")
OUT = os.path.join(paths.OUT, "mscxrt/manifest.jsonl")
FINDINGS = ["consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"]


def rep_path(subj, sid): return f"{REP}/p{str(subj)[:2]}/p{subj}/s{sid}.txt"
def img_path(dicom): return f"{IMG}/{dicom}.jpg"


def read_report(p):
    try:
        return open(p).read().strip()
    except OSError:
        return ""


def main():
    df = pd.read_csv(CSV)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0; skipped = 0
    with open(OUT, "w") as f:
        for r in df.itertuples():
            labels = {fd: getattr(r, f"{fd}_progression") for fd in FINDINGS
                      if isinstance(getattr(r, f"{fd}_progression"), str)}
            if not labels:
                continue
            cip = img_path(r.dicom_id)
            if not os.path.exists(cip):
                skipped += 1; continue
            rec = {
                "pair_id": f"{r.previous_study_id}_{r.study_id}",
                "subject_id": int(r.subject_id),
                "current_study_id": int(r.study_id), "prior_study_id": int(r.previous_study_id),
                "current_dicom": r.dicom_id, "prior_dicom": r.previous_dicom_id,
                "current_image": cip,
                "prior_report": read_report(rep_path(r.subject_id, r.previous_study_id)),
                "current_report": read_report(rep_path(r.subject_id, r.study_id)),  # for reference/upper-bound only
                "labels": labels,
            }
            f.write(json.dumps(rec) + "\n"); n += 1
    print(f"wrote {OUT}  pairs={n}  skipped(no image)={skipped}")
    # label distribution
    import collections
    dist = collections.defaultdict(collections.Counter)
    for line in open(OUT):
        for fd, c in json.loads(line)["labels"].items():
            dist[fd][c] += 1
    print("\nper-finding labeled instances (improving/stable/worsening):")
    for fd in FINDINGS:
        d = dist[fd]
        print(f"  {fd:16s} n={sum(d.values()):4d}  {dict(d)}")


if __name__ == "__main__":
    main()
