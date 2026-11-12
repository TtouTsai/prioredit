#!/usr/bin/env python
"""PriorEdit's rule editor -- paper Sec. 6.2, Eq. 2.

Given the prior report text and the image-only draft, decide each finding's current state and
attach a prior-grounded temporal tag. The prior is consulted only here; it never enters the
generator.

  confidence_state()  Eq. 2. The draft is authoritative where explicit; where the draft is silent
                      and the prior reports the finding present, the calibrated confidence
                      c_f(I) decides recover (pos, c >= tau_pos) / resolved (neg, c <= tau_neg) /
                      abstain (none).
  edit_inplace()      maps (ext(R)_f, s_f) to a tag (new / resolved / unchanged, or an
                      increased/decreased note on an unambiguous severity jump) and merges it into
                      the draft sentence in place; adds a grounded sentence only where confidence
                      recovered a finding the draft left silent. The draft is never contradicted
                      and the report is not duplicated.

Adopted configuration (reproduces the paper's PriorEdit generations byte-for-byte):
  --confidence data/finding_confidence.json --thresholds configs/confidence_thresholds.json \
  --inplace --high_conf --keep_unchanged --sev_jump 2

Input drafts json {study_id: draft_text}; output {study_id: edited_text}.
  python rule_editor.py --drafts <draft json> --ann <annotation with prior report> --out <json> [flags]
"""
import os as _os, sys as _sys
_scripts = _os.path.dirname(_os.path.abspath(__file__))
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402,F401  (also puts scripts/* groups on sys.path)
import json, argparse, os, re, sys
from state_extractor import extract, official_extract, load_official

# extraction backend: default regex extract(); --official switches to the CheXpert labeler (state) +
# our SEV lexicon (severity). Wrapped in a pointer so all call sites switch together.
_EXTRACT = extract

# full CheXbert pathology set (GT's temporal statements are often about opacity/nodule -> must cover
# them). Broad lexicons are safe for in-place (only modify sentences the base already states); for the
# inject/union path they can over-fire, so keep the greedy-authoritative + threshold guard.
NAME = {"consolidation": "consolidation", "edema": "pulmonary edema",
        "pleural_effusion": "pleural effusion", "pneumonia": "pneumonia", "pneumothorax": "pneumothorax",
        "atelectasis": "atelectasis", "cardiomegaly": "cardiomegaly", "lung_opacity": "opacity",
        "lung_lesion": "nodule", "enlarged_cardiomediastinum": "mediastinal enlargement",
        "pleural_other": "pleural thickening", "fracture": "fracture"}
EDIT_FINDINGS = list(NAME.keys())

# a prior that provides no basis for temporal comparison -> do NOT insert any temporal clause
NO_PRIOR = re.compile(r"no prior|no old films?|not available for comparison|no comparison|no previous",
                      re.I)


def has_prior(prior_text):
    return bool(prior_text and prior_text.strip() and prior_text.strip() != "s"
                and not NO_PRIOR.search(prior_text))


def union_state(greedy_draft, samples, k=2):
    """current per-finding state. The greedy draft is AUTHORITATIVE on its explicit statements (never
    contradict it); the temperature samples only FILL greedy's silence:
      greedy pos  -> pos ; greedy neg -> neg (samples cannot override) ; greedy silent -> pos iff >=k
      samples say pos, else none.
    This avoids self-contradiction (injecting a finding the draft negated) and the false injections that
    come from a few noisy samples overriding a confident greedy negation."""
    g = _EXTRACT(greedy_draft)
    exs = [_EXTRACT(s) for s in samples]
    out = {}
    for fd in EDIT_FINDINGS:
        gs = g[fd][0]
        if gs == "pos":
            out[fd] = "pos"                                  # base authoritative on its explicit statements
        elif gs == "neg":
            out[fd] = "neg"                                  # (samples cannot override -> no contradiction)
        else:                                                # base silent -> MAJORITY VOTE among samples
            pos_v = sum(e[fd][0] == "pos" for e in exs)
            neg_v = sum(e[fd][0] == "neg" for e in exs)
            if pos_v >= k and pos_v > neg_v:
                out[fd] = "pos"
            elif neg_v >= k and neg_v >= pos_v:
                out[fd] = "neg"
            else:
                out[fd] = "none"                             # tie / insufficient evidence
    return out


def confidence_state(greedy_draft, prior_text, conf, thresholds):
    """Coverage completion via the model's own per-finding confidence (probe_confidence.py), a drop-in
    alternative to union_state's resampling vote. Greedy draft stays AUTHORITATIVE on its explicit
    statements (pos/neg). Scope = findings the draft is SILENT on AND the prior states positive:
      conf > tau_pos -> pos  (recover a persistent finding the draft missed -> editor writes
                              'unchanged'/'increased' vs the prior)
      conf < tau_neg -> neg  (confident absence -> editor writes 'resolved')
      else -> none.
    Findings the prior does not state positive stay none (no temporal basis). Note: prior=neg + draft
    silent (would give 'new') was tried and dropped -- it barely moved temporal (+0.006) but raised
    hallucination, because prior-absent findings rarely actually appear now."""
    g = _EXTRACT(greedy_draft)
    p = _EXTRACT(prior_text)
    conf = conf or {}
    out = {}
    for fd in EDIT_FINDINGS:
        gs = g[fd][0]
        if gs == "pos":
            out[fd] = "pos"
        elif gs == "neg":
            out[fd] = "neg"
        else:                                            # draft silent
            th = thresholds.get(fd, {}); c = conf.get(fd)
            if p[fd][0] == "pos" and c is not None:      # prior=pos, draft missed it
                if th.get("open_pos") and th.get("tau_pos") is not None and c > th["tau_pos"]:
                    out[fd] = "pos"
                elif th.get("open_neg") and th.get("tau_neg") is not None and c < th["tau_neg"]:
                    out[fd] = "neg"
                else:
                    out[fd] = "none"
            else:
                out[fd] = "none"
    return out


TAG = {"unchanged": "unchanged from prior", "new": "new from prior", "resolved": "now resolved",
       "increased": "increased from prior", "decreased": "decreased from prior"}


def _tag_for(bstate, pstate, csev=None, psev=None):
    """verify step: decide the temporal tag from base vs prior. For a PERSISTENT finding, use severity
    evidence to output increased/decreased (matches GT's severity vocabulary) when both sides have a
    severity level; otherwise default to 'unchanged' (uncertain but usually right). grounding: prior
    SILENCE is not evidence -> only claim 'new' when prior EXPLICITLY negates."""
    if bstate == "pos" and pstate == "pos":                # persistent
        if csev is not None and psev is not None:
            return "increased" if csev > psev else "decreased" if csev < psev else "unchanged"
        return "unchanged"
    if bstate == "pos" and pstate == "neg":
        return "new"
    if bstate == "neg" and pstate == "pos":
        return "resolved"
    return None


def _tag_for_hc(bstate, pstate, csev=None, psev=None, sev_jump=2, keep_unchanged=False):
    """HIGH-CONFIDENCE tag policy. MS-CXR-T showed absolute-severity comparison cannot recover fine
    direction, so we only assert a directional CHANGE when it is reliable from prior text + draft:
      absent -> present            -> 'new'        (presence change)
      present -> absent            -> 'resolved'   (presence change)
      present -> present w/ a clear severity JUMP (|csev-psev| >= sev_jump, e.g. small->large) -> inc/dec
      present -> present otherwise -> 'unchanged' if keep_unchanged else None
    'unchanged' on a genuinely-persistent finding is presence-grounded (both sides present), NOT a
    fabricated direction -> keep_unchanged=True keeps it (drives Temporal-F1); only the low-confidence
    inc/dec flips are gated out. prior SILENCE is never evidence (callers pass explicit-extract pstate)."""
    if bstate == "pos" and pstate == "neg":
        return "new"
    if bstate == "neg" and pstate == "pos":
        return "resolved"
    if bstate == "pos" and pstate == "pos":
        if csev is not None and psev is not None:
            if csev - psev >= sev_jump:
                return "increased"
            if psev - csev >= sev_jump:
                return "decreased"
        return "unchanged" if keep_unchanged else None
    return None


def edit_inplace(prior_text, draft, cur_states=None, high_conf=False, sev_jump=2, keep_unchanged=False):
    """MODIFY the base draft's existing finding sentences into temporal ones (no injection). If
    cur_states (a resampling UNION) is given, also FILL findings the base was silent on but the union
    confirms (real greedy misses) -> resampling recall + prior temporal, without contradiction (greedy
    authoritative). high_conf=True uses the conservative _tag_for_hc policy (presence-change + clear
    severity jump only; no low-confidence 'unchanged')."""
    if not has_prior(prior_text):
        return draft
    tagger = (lambda b, p, c, s: _tag_for_hc(b, p, c, s, sev_jump, keep_unchanged)) if high_conf else _tag_for
    p = _EXTRACT(prior_text)
    body = re.sub(r"^\s*finding\s*:\s*", "", draft.strip(), flags=re.I)
    sents = [s.strip().rstrip(".").strip() for s in body.split(" . ") if s.strip().rstrip(".").strip()]
    cbody = _EXTRACT(draft)   # states from the FULL draft (matches labeled text; finding-prefix has no findings)
    # which EDIT_FINDINGS each draft sentence mentions (regex, only for PLACEMENT of the temporal tag)
    sent_finds = [{fd for fd in EDIT_FINDINGS if extract(s)[fd][0] in ("pos", "neg")} for s in sents]
    done, appended = set(), []
    # for each finding the draft STATES: attach the temporal tag onto the draft sentence in place when that
    # sentence mentions only this finding; the tag NAMES the finding ("{name} is {tag}") so the temporal claim
    # is self-identifying. A finding sharing its sentence with others (ambiguous attribution) or spread over
    # several sentences becomes its own finding-bound sentence. Either way each temporal claim lives in a
    # single-finding clause, so the metric parser never has to split a change across co-located findings.
    for fd in EDIT_FINDINGS:
        if cbody[fd][0] == "none":
            continue
        tag = tagger(cbody[fd][0], p[fd][0], cbody[fd][1], p[fd][1])
        done.add(fd)
        if not tag:
            continue
        idxs = [i for i, sf in enumerate(sent_finds) if fd in sf]
        if len(idxs) == 1 and sent_finds[idxs[0]] == {fd}:          # sole finding in its sentence -> in-place, NAMED
            sents[idxs[0]] = f"{sents[idxs[0]]}, {NAME[fd]} is {TAG[tag]}"
        else:                                                       # ambiguous -> own finding-bound sentence
            appended.append(f"{NAME[fd]} is {TAG[tag]}")
    # coverage fill for findings the draft is SILENT on (prior=pos):
    #   cur_states=='pos' -> recovered persistent (greedy miss)   -> "there is {name} unchanged"
    #   cur_states=='neg' -> confident current absence (tau_neg)  -> "no {name}, now resolved"
    if cur_states is not None:
        for fd in EDIT_FINDINGS:
            if fd in done or cbody[fd][0] != "none":
                continue
            if cur_states[fd] == "pos":
                tag = tagger("pos", p[fd][0], None, p[fd][1])
                appended.append(f"there is {NAME[fd]}" + (f" {TAG[tag]}" if tag else ""))
            elif cur_states[fd] == "neg":
                tag = tagger("neg", p[fd][0], None, p[fd][1])   # _tag_for_hc(neg,pos) -> 'resolved'
                if tag:
                    appended.append(f"no {NAME[fd]}, {TAG[tag]}")
    return "finding : " + " . ".join(sents + appended) + " ."


def edit(prior_text, draft, no_persistent=False, cur_states=None):
    if not has_prior(prior_text):     # fix: no prior basis -> no temporal claims (input-agnostic halluc)
        return draft
    p = _EXTRACT(prior_text)
    c = _EXTRACT(draft) if cur_states is None else None
    clauses = []
    for fd in EDIT_FINDINGS:
        pp = p[fd][0] == "pos"
        cstate = cur_states[fd] if cur_states is not None else c[fd][0]
        cc = cstate == "pos"
        # union current-presence is reliable -> relaxed symmetric change logic (no explicit-neg required)
        if not pp and cc:                                     # appeared: prior not present, current present
            clauses.append(f"interval development of {NAME[fd]}")
        elif pp and not cc:                                   # resolved: prior present, current absent
            clauses.append(f"previously noted {NAME[fd]} has resolved")
        elif pp and cc and not no_persistent:                 # persistent
            clauses.append(f"{NAME[fd]} is unchanged")
    if not clauses:
        return draft
    body = draft.rstrip().rstrip(".").rstrip()
    return body + " . " + " . ".join(clauses) + " ."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", required=True); ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--no_persistent", action="store_true")
    ap.add_argument("--samples", help="resampled drafts json {id:[N texts]} -> union current-presence")
    ap.add_argument("--vote_k", type=int, default=2, help="min temperature samples to confirm a greedy-missed finding")
    ap.add_argument("--inplace", action="store_true", help="rewrite base sentences into temporal ones (no injection)")
    ap.add_argument("--high_conf", action="store_true", help="conservative: presence-change + clear severity jump only")
    ap.add_argument("--sev_jump", type=int, default=2, help="min severity-level jump to assert increased/decreased")
    ap.add_argument("--keep_unchanged", action="store_true", help="keep presence-grounded 'unchanged' for persistent findings (Temporal-F1)")
    ap.add_argument("--official", action="store_true", help="use official CheXpert labeler for state (severity still from SEV lexicon)")
    ap.add_argument("--confidence", help="per-finding confidence json {id:{finding:logP(pos)-logP(neg)}} -> confidence coverage, Eq. 2 (replaces resampling vote)")
    ap.add_argument("--thresholds", help="per-finding threshold json (configs/confidence_thresholds.json)")
    a = ap.parse_args()
    if a.official:
        global _EXTRACT
        load_official(); _EXTRACT = official_extract
    drafts = json.load(open(a.drafts))
    samples = json.load(open(a.samples)) if a.samples else None
    conf_map = json.load(open(a.confidence)) if a.confidence else None
    thresholds = json.load(open(a.thresholds)) if a.thresholds else {}
    priors = {}
    for r in json.load(open(a.ann))["test"]:
        lf = r.get("last_finding", "")
        pv = "" if lf.strip() == "s" else lf
        priors[str(r["id"])] = pv           # test_result.json is keyed by 'id' (dicom)
        priors[str(r["study_id"])] = pv     # also allow study_id keys
    out = {}
    n_edited = 0
    for sid, d in drafts.items():
        dt = d[0] if isinstance(d, list) else d
        cur = None
        if conf_map is not None and str(sid) in conf_map:        # confidence coverage (Eq. 2)
            cur = confidence_state(dt, priors.get(str(sid), ""), conf_map[str(sid)], thresholds)
        elif samples is not None and str(sid) in samples:
            cur = union_state(dt, samples[str(sid)], k=a.vote_k)  # greedy-authoritative + threshold fill
        if a.inplace:
            e = edit_inplace(priors.get(str(sid), ""), dt, cur_states=cur,
                             high_conf=a.high_conf, sev_jump=a.sev_jump, keep_unchanged=a.keep_unchanged)
        else:
            e = edit(priors.get(str(sid), ""), dt, a.no_persistent, cur_states=cur)
        out[sid] = e
        if e != dt:
            n_edited += 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"edited {n_edited}/{len(out)} drafts -> {a.out}")


if __name__ == "__main__":
    main()
