#!/usr/bin/env python
"""Per-finding state extractor ext(.)_f -- paper Sec. 6.1 and App. B.

Reads a report and returns, per finding, one of {pos, neg, none} = present / explicitly absent /
unmentioned. For the five MS-CXR-T findings the mention lexicon is aligned to the official Stanford
CheXpert labeler's phrase lists, so the extractor implements the authoritative label definitions
rather than a hand-tuned lexicon; the remaining findings use hand-written mention patterns.
Negation uses NegBio-style cues ("no", "without", "clear", "resolved"). The hand-written extractor has
no uncertain state; the official-labeler variant below maps uncertain to present.

Also provides the severity ordinal (trace/small/mild < moderate < large/severe) used only for the
severity-jump tag, and comparison logic that turns a (prior, current) state pair into the two
temporal layers:
  presence-change: appeared / resolved / persistent
  severity:        improving / stable / worsening   (appeared -> worsening, resolved -> improving)

Used by the rule editor (to read the prior and the draft), by TCA (to bind a claim to a finding),
and by the MS-CXR-T direction probe. `official_extract()` / `load_official()` swap in the official
labeler's own states instead of the regex.

Usage (score on the gold current report = the verifier-logic ceiling):
  python state_extractor.py --current gold             # uses the manifest's current_report
  python state_extractor.py --current <drafts.json>    # {study_id or pair_id: draft_text}
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import json, re, argparse, collections, os, subprocess

MANIFEST = os.path.join(paths.OUT, "mscxrt/manifest.jsonl")
OUTDIR = os.path.join(paths.OUT, "mscxrt")
SCORER = os.path.join(paths.SCRIPTS, "direction_probe", "score_mscxrt.py")
PY = paths.PY_LABEL
FINDINGS = ["consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"]  # MS-CXR-T subset

FIND_KW = {
    # MS-CXR-T 5. Phrase lists ALIGNED to the official Stanford CheXpert labeler mention files
    # (github.com/stanfordmlgroup/chexpert-labeler, phrases/mention/*.txt) so the extractor is an
    # implementation of the authoritative label definitions rather than a hand-tuned lexicon. Note the
    # official edema list has NO "fluid overload" (removed); it does have chf/heart failure/vascular
    # prominence/indistinctness (added).
    "consolidation": [r"consolidat\w*"],
    "edema": [r"edema", r"heart failure", r"\bchf\b", r"vascular congestion", r"pulmonary congestion",
              r"vascular prominence", r"indistinctness"],
    "pleural_effusion": [r"pleural effusion", r"effusion", r"pleural fluid"],
    "pneumonia": [r"pneumonia", r"infectio\w*", r"infection"],
    "pneumothorax": [r"pneumothorax", r"pneumothoraces"],
    # rest of CheXbert-14 (pathologies + devices) for temporal editing on general eval
    "cardiomegaly": [r"cardiomegaly", r"enlarged (?:cardiac|heart)", r"heart.{0,15}enlarg\w*",
                     r"cardiac silhouette.{0,15}enlarg\w*", r"enlarged cardiac silhouette"],
    "enlarged_cardiomediastinum": [r"widen\w* .{0,10}mediastin\w*", r"mediastinal widening",
                                   r"enlarged cardiomediastin\w*", r"mediastin\w*.{0,10}enlarg\w*"],
    "lung_opacity": [r"opacit\w*", r"opacification"],
    "lung_lesion": [r"nodul\w*", r"\bmass\b", r"\blesion\b"],
    "atelectasis": [r"atelecta\w*", r"collapse"],
    "pleural_other": [r"pleural thickening", r"pleural scarring", r"pleural.{0,10}(?:thick|scar)\w*"],
    "fracture": [r"fracture\w*"],
    "support_devices": [r"\btube\b", r"catheter", r"\bpicc\b", r"pacemaker", r"pacer",
                        r"\bport\b", r"sternotomy", r"\bstent\b"],
}
FINDINGS_14 = list(FIND_KW.keys())  # all findings the extractor/editor cover
SEV = {"trace": 1, "minimal": 1, "tiny": 1, "small": 2, "mild": 2, "slight": 2, "little": 2,
       "moderate": 3, "significant": 3, "substantial": 3.5, "large": 4, "severe": 4, "extensive": 4,
       "sizable": 3.5, "gross": 4, "massive": 4, "worsening": None, "layering": 2}
NEG = re.compile(r"\b(no|without|free of|clear of|absence of|resolution of|resolved|not|never|"
                 r"unremarkable|negative for|rule out|ruled out)\b")


def sentences(t):
    return [s.strip() for s in re.split(r"[.;\n]", (t or "").lower()) if s.strip()]


def sev_in(window):
    lv = [SEV[w] for w in re.findall(r"[a-z-]+", window) if w in SEV and SEV[w] is not None]
    return max(lv) if lv else None


def extract(text):
    """finding -> (state, severity). state in {'pos','neg','none'}; severity int or None."""
    out = {}
    sents = sentences(text)
    for fd, kws in FIND_KW.items():
        state, sev = "none", None
        for s in sents:
            for kw in kws:
                m = re.search(kw, s)
                if not m:
                    continue
                # negation: a NEG cue anywhere before the keyword in this sentence
                pre = s[:m.start()]
                neg = bool(NEG.search(pre)) or bool(re.match(r"\s*no\b", s))
                if neg:
                    if state == "none":
                        state = "neg"
                else:
                    state = "pos"
                    sv = sev_in(s)
                    if sv and (sev is None or sv > sev):
                        sev = sv
            if state == "pos":
                break
        out[fd] = (state, sev)
    return out


# ---- official CheXpert labeler as the extraction layer (state) + our SEV lexicon (severity) ----
import hashlib, json as _json, os as _os  # noqa: E402

_OFFICIAL_STORE = None
_OFFICIAL_PATH = os.path.join(paths.LABELS, "official_labels_by_hash.json")


def _clean_for_hash(t):
    # must match chexpert_label._clean so hashes line up
    return " ".join((t or "").replace("\n", " ").split()) or "no report"


def load_official(path=None):
    """Load the {sha1(clean_text): {finding: pos|neg|unc|none}} store (data/labels/official_labels_by_hash.json): the official labeler's output keyed by
    the hash of the cleaned text."""
    global _OFFICIAL_STORE
    p = path or _OFFICIAL_PATH
    _OFFICIAL_STORE = _json.load(open(p)) if _os.path.exists(p) else {}
    return _OFFICIAL_STORE


def official_extract(text):
    """Drop-in for extract() but state comes from the OFFICIAL CheXpert labeler (mention lists + NegBio);
    severity is layered from our SEV lexicon on the text. uncertain(official) -> 'pos' (present, matching
    the paper's binarization). Falls back to the regex extract() when a text is not in the store."""
    if _OFFICIAL_STORE is None:
        load_official()
    h = hashlib.sha1(_clean_for_hash(text).encode()).hexdigest()
    lab = _OFFICIAL_STORE.get(h)
    if lab is None:
        return extract(text)                         # unlabeled -> regex fallback (keeps pipeline robust)
    reg = extract(text)                              # reuse regex only for the severity number
    out = {}
    for fd in FIND_KW:
        st = lab.get(fd, "none")
        state = "pos" if st in ("pos", "unc") else "neg" if st == "neg" else "none"
        sev = reg[fd][1] if state == "pos" else None
        out[fd] = (state, sev)
    return out


def verify(prior_text, cur_text):
    p, c = extract(prior_text), extract(cur_text)
    prog, change = {}, {}
    for fd in FINDINGS:
        ps, psev = p[fd]; cs, csev = c[fd]
        p_present = ps == "pos"
        c_present = cs == "pos"
        # presence-change layer
        if p_present and c_present:
            change[fd] = "persistent"
        elif p_present and not c_present:
            change[fd] = "resolved"
        elif not p_present and c_present:
            change[fd] = "appeared"
        else:
            change[fd] = "persistent"   # neither clearly present; default (rare for labeled instances)
        # severity/progression layer (MS-CXR-T)
        if change[fd] == "resolved":
            prog[fd] = "improving"
        elif change[fd] == "appeared":
            prog[fd] = "worsening"
        else:  # persistent -> compare severity
            if psev is not None and csev is not None:
                prog[fd] = "improving" if csev < psev else ("worsening" if csev > psev else "stable")
            else:
                prog[fd] = "stable"     # severity unknown -> assume stable
    return prog, change


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--current", default="gold"); args = ap.parse_args()
    manifest = [json.loads(l) for l in open(MANIFEST)]
    drafts = None
    if args.current != "gold":
        drafts = json.load(open(args.current))
    pred_prog, pred_change = collections.defaultdict(dict), collections.defaultdict(dict)
    miss = 0
    for r in manifest:
        prior = r["prior_report"]
        if drafts is None:
            cur = r["current_report"]
        else:
            cur = drafts.get(str(r["current_study_id"])) or drafts.get(r["pair_id"]) or ""
            if not cur:
                miss += 1
        prog, change = verify(prior, cur)
        for fd in r["labels"]:
            pred_prog[r["pair_id"]][fd] = prog[fd]
            pred_change[r["pair_id"]][fd] = change[fd]
    tag = "gold" if drafts is None else os.path.basename(args.current).replace(".json", "")
    pp = f"{OUTDIR}/pred_verifier_{tag}.json"
    json.dump(pred_prog, open(pp, "w"))
    json.dump(pred_change, open(f"{OUTDIR}/pred_verifierchange_{tag}.json", "w"))
    if miss:
        print(f"(warning: {miss} pairs had no draft)")
    print(f"### v0 verifier severity/progression (current={tag}) vs MS-CXR-T ###")
    subprocess.run([PY, SCORER, "--pred", pp], check=True)


if __name__ == "__main__":
    main()
