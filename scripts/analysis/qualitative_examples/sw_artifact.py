#!/usr/bin/env python
"""Flag rows where sw's asserted class is a REGEX ARTIFACT rather than a real sw error.

temporal_accuracy.NEW contains `appear\\w*`, and asserted_transitions() tests the 'new' branch
BEFORE the persistent branch. So a sentence like

    "the cardiac ... contours appear unchanged including cardiomegaly"

matches NEW via 'appear' and is scored 'new' even though it literally says *unchanged*. Those rows
must not be shown as sw failures in the paper.

This module does NOT modify temporal_accuracy.py. It re-derives, for a given (report, finding), the
sentence that drove sw's label and reports whether that label survives once the ambiguous 'appear'
cue is required to yield to an explicit persistence word.

  verdict 'genuine'  : label stands on an unambiguous cue -> usable as a real sw error
  verdict 'weak'     : rests only on a copular 'appear' -> judgement call, kept out of the main table
  verdict 'artifact' : 'appear*' in a sentence that states persistence / no-change, or the noun
                       "appearance" -> definitely NOT an sw error
"""
import os as _os, sys as _sys
_scripts = _os.path.dirname(_os.path.abspath(__file__))
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402,F401  (also puts scripts/* groups on sys.path)
import os, re, sys


from temporal_accuracy import NEW, RESOLVED, PERSIST, NO_NEW, NOT_RESOLVED, FIND  # noqa: E402
from state_extractor import extract  # noqa: E402

APPEAR = re.compile(r"\bappear\w*\b", re.I)
# 'appear' is only a NEW cue when it means "has become apparent". These two uses are NOT temporal:
#   copular  -- "the lungs appear clear", "opacities appear more consolidated" (describes a state)
#   nominal  -- "little change in the APPEARANCE of ...", "similar in appearance" (a noun, and here
#               it is usually attached to an explicit no-change phrase)
APPEAR_NOMINAL = re.compile(r"\bappearances?\b", re.I)
NO_CHANGE = re.compile(r"\b(?:little|no|without)\s+(?:overall\s+|significant\s+)?change\b", re.I)
# NEW minus the ambiguous 'appear*' cue
NEW_STRICT = re.compile(
    r"\b(new|newly|interval development|developed|develop|"
    r"now (?:seen|present|noted|evident)|has developed)\b", re.I)


def trigger_sentence(report, finding, label):
    """The first sentence that would have produced `label` for `finding` under the shipped logic."""
    for s in re.split(r"[.;\n]", (report or "")):
        s_new = bool(NEW.search(s)) and not NO_NEW.search(s)
        s_res = bool(RESOLVED.search(s)) and not NOT_RESOLVED.search(s)
        s_per = bool(PERSIST.search(s))
        if not (s_new or s_res or s_per):
            continue
        st = extract(s)[finding][0]
        if st == "none":
            continue
        if label == "new" and s_new and st == "pos":
            return s.strip()
        if label == "resolved" and s_res:
            return s.strip()
        if label == "persistent" and s_per and st == "pos":
            return s.strip()
    return None


def classify(report, finding, label):
    """-> (verdict, trigger_sentence, reason)."""
    s = trigger_sentence(report, finding, label)
    if s is None:
        return "unknown", "N/A", "no triggering sentence recovered"
    if label != "new":
        return "genuine", s, "label does not depend on the ambiguous 'appear' cue"
    strict = bool(NEW_STRICT.search(s))
    persist = bool(PERSIST.search(s))
    appear = bool(APPEAR.search(s))
    if strict:
        return "genuine", s, "explicit new-cue present"
    if appear and persist:
        return "artifact", s, "'appear*' only, sentence also states persistence"
    if appear and NO_CHANGE.search(s):
        return "artifact", s, "'appear*' only, sentence explicitly states no change"
    if appear and APPEAR_NOMINAL.search(s):
        return "artifact", s, "'appearance' is a noun here, not a temporal cue"
    if appear:
        return "weak", s, "'appear*' as copula (state description), not an explicit new-cue"
    return "genuine", s, "non-appear new-cue"
