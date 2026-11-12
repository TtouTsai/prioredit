#!/usr/bin/env python
"""Run the OFFICIAL Stanford CheXpert labeler (dockerized) on a set of reports and return, per report,
the 14 CheXpert labels as state in {pos, neg, unc, none}. This is the authoritative rule labeler
(mention phrase lists + NegBio negation/uncertainty), used to replace both (a) the hallucination GT and
(b) our hand-written rule extractor.

The labeler container is built from https://github.com/stanfordmlgroup/chexpert-labeler and tagged
`chexpert-labeler`. Output CSV columns: Reports + 14 CATEGORIES; cell 1.0/0.0/-1.0/blank =
positive/negative/uncertain/not-mentioned.

CLI:
  python chexpert_label.py --in reports.json --out labels.json     # {id: text} -> {id: {finding: state}}
  python chexpert_label.py --in reports.jsonl ...                  # or a plain list
Reports are newline-joined into one CSV (one report per row, quoted) and mounted into the container.
"""
import json, argparse, os, subprocess, csv, sys, tempfile

IMAGE = "chexpert-labeler"
# official CATEGORIES order (constants/__init__.py)
CATEGORIES = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity",
              "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
              "Pleural Other", "Fracture", "Support Devices"]
# map official category -> our finding key (state_extractor FIND_KW keys)
CAT2FD = {"Enlarged Cardiomediastinum": "enlarged_cardiomediastinum", "Cardiomegaly": "cardiomegaly",
          "Lung Lesion": "lung_lesion", "Lung Opacity": "lung_opacity", "Edema": "edema",
          "Consolidation": "consolidation", "Pneumonia": "pneumonia", "Atelectasis": "atelectasis",
          "Pneumothorax": "pneumothorax", "Pleural Effusion": "pleural_effusion",
          "Pleural Other": "pleural_other", "Fracture": "fracture", "Support Devices": "support_devices"}


def _clean(t):
    return " ".join((t or "").replace("\n", " ").split()) or "no report"


def run_labeler(texts, workdir):
    """texts: list[str] -> list[dict finding->state]. One docker invocation."""
    in_csv = os.path.join(workdir, "reports.csv")
    out_csv = os.path.join(workdir, "labeled.csv")
    # labeler reads a header-less CSV, one report per row, first column = report
    with open(in_csv, "w", newline="") as f:
        w = csv.writer(f)
        for t in texts:
            w.writerow([_clean(t)])
    cmd = ["docker", "run", "--rm", "-v", f"{workdir}:/data", IMAGE,
           "python", "label.py", "--reports_path", "/data/reports.csv",
           "--output_path", "/data/labeled.csv", "--verbose"]
    subprocess.run(cmd, check=True)
    out = []
    with open(out_csv) as f:
        for row in csv.DictReader(f):
            st = {}
            for cat, fd in CAT2FD.items():
                v = row.get(cat, "").strip()
                st[fd] = ("pos" if v == "1.0" else "neg" if v == "0.0"
                          else "unc" if v == "-1.0" else "none")
            out.append(st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="json {id:text} or list, or jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    raw = open(a.inp).read().strip()
    if a.inp.endswith(".jsonl"):
        data = [json.loads(l) for l in raw.splitlines()]
        ids = [str(i) for i in range(len(data))]
        texts = [d if isinstance(d, str) else d.get("text", "") for d in data]
    else:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            ids = list(obj.keys())
            texts = [v[0] if isinstance(v, list) else v for v in obj.values()]
        else:
            ids = [str(i) for i in range(len(obj))]
            texts = [v[0] if isinstance(v, list) else v for v in obj]
    wd = a.workdir or tempfile.mkdtemp(prefix="chexpert_")
    os.makedirs(wd, exist_ok=True)
    labels = run_labeler(texts, wd)
    out = {i: lab for i, lab in zip(ids, labels)}
    json.dump(out, open(a.out, "w"))
    print(f"labeled {len(out)} reports -> {a.out}")


if __name__ == "__main__":
    main()
