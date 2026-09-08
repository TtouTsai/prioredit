#!/usr/bin/env python
"""Verbatim sentence extraction for the Figure-2 candidate tables.

state_extractor.sentences() LOWERCASES and strips, so it must never be used to produce the text that
goes into the paper. Here we split on the same delimiters ([.;\n]) but keep the ORIGINAL slice of
the source string, so every emitted sentence is byte-identical to the report it came from
(no rewriting, no truncation, no added punctuation).

Matching a finding to a sentence reuses state_extractor.extract() on the lowercased sentence (the
extractor's own convention) while the emitted text stays the untouched original.
"""
import os as _os, sys as _sys
_scripts = _os.path.dirname(_os.path.abspath(__file__))
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402,F401  (also puts scripts/* groups on sys.path)
import os, re, sys


from state_extractor import extract  # noqa: E402

_SPLIT = re.compile(r"[.;\n]")


def split_verbatim(text):
    """Split into (start, end, verbatim_text) keeping the exact original substring.

    Only leading/trailing whitespace is trimmed from the span -- inner characters, casing and
    punctuation are preserved byte-for-byte."""
    if not text:
        return []
    out, pos = [], 0
    for m in _SPLIT.finditer(text):
        seg = text[pos:m.start()]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            s = pos + lead
            e = pos + len(seg.rstrip())
            out.append((s, e, text[s:e]))
        pos = m.end()
    seg = text[pos:]
    if seg.strip():
        lead = len(seg) - len(seg.lstrip())
        s = pos + lead
        e = pos + len(seg.rstrip())
        out.append((s, e, text[s:e]))
    return out


def sentences_for_finding(text, finding, drop_header=True):
    """All verbatim sentences in `text` that state `finding` (pos or neg).

    drop_header removes a bare 'finding :' / 'Finding:' lead-in, which carries no finding and would
    otherwise be emitted as a sentence."""
    hits = []
    for s, e, verb in split_verbatim(text or ""):
        core = re.sub(r"^\s*finding\s*:?\s*", "", verb, flags=re.I)
        if drop_header and not core.strip():
            continue
        st = extract(verb.lower())[finding][0]
        if st in ("pos", "neg"):
            hits.append(verb)
    return hits


def joined(text, finding, sep=" ||| "):
    """Verbatim sentences joined for a single CSV cell; 'unmentioned' when the finding is absent."""
    h = sentences_for_finding(text, finding)
    return sep.join(h) if h else "unmentioned"


def wordcount(s):
    return 0 if s in ("", "unmentioned", "N/A") else len(s.split())
