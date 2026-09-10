#!/usr/bin/env python
"""Independent-parser robustness (paper App. E): extract the ASSERTED temporal transition with a
local chat LLM instead of the regex parser, and cache it for scoring.

Structural twin of scripts/analysis/independent_labeler.py: that swaps the LABELER on TCA's *true*
side and reuses the unchanged temporal_assessment.score(); this swaps the PARSER on the *asserted*
side and likewise reuses the unchanged score(). Nothing in the pipeline is modified -- this only
writes a cache of per-finding asserted transitions ({id: {finding: new|resolved|persistent|none}}),
which llm_parser_eval.py feeds into the unchanged scorer. The cache is text-free and shipped under
data/llm_parser/, so App. E can be re-scored without re-running any model.

Local Ollama only: MIMIC-CXR report text is under a PhysioNet DUA and must not leave the machine.
Temperature 0, thinking disabled, strict JSON; the model digest and prompt hash are pinned in the
meta file for reproducibility. The paper runs two models:
  --model qwen3:30b-a3b            (general-purpose)
  --model symptoma/medgemma3:27b   (instruction-tuned medical)

  python llm_parser_extract.py --system gtself --n 0 --tag qwen3
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots from PRIOREDIT_* env vars)
import json, os, re, sys, time, argparse, hashlib, random, urllib.request

RG = paths.LLMRG4
OUTDIR = os.environ.get("PRIOREDIT_LLM_PARSER", os.path.join(paths.ROOT, "data", "llm_parser"))
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# identical to temporal_assessment.FIND (state_extractor.FIND_KW minus support_devices)
FIND = ["consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax", "cardiomegaly",
        "enlarged_cardiomediastinum", "lung_opacity", "lung_lesion", "atelectasis",
        "pleural_other", "fracture"]
CLASSES = {"new", "resolved", "persistent", "none"}

# The four category definitions deliberately mirror the regex parser's NEW / RESOLVED / PERSIST
# semantics (temporal_assessment.py) so the comparison is parser-vs-parser, not
# definition-vs-definition. In particular 'persistent' must include increased/decreased/improved/
# worsened, because those imply the finding was present at BOTH timepoints.
PROMPT = """You are a precise information extractor for radiology reports.

Given the report text below, for EACH finding in the list, decide what the report
EXPLICITLY ASSERTS about how that finding changed relative to a prior study:

  "new"        - the report explicitly states this finding is new / newly appeared /
                 developed in the interval / is now seen or now present.
  "resolved"   - the report explicitly states this finding has resolved / cleared /
                 is no longer seen / has been removed.
  "persistent" - the report explicitly compares to a prior and the finding is still
                 present: unchanged, stable, persistent, again seen, redemonstrated,
                 previously noted and still there, OR changed in degree while still
                 present (increased, decreased, improved, worsened, larger, smaller).
  "none"       - anything else.

CRITICAL RULES:
1. Report ONLY explicit comparisons that appear in the text. Do NOT infer, guess, or
   use medical knowledge about what probably changed. If the report does not explicitly
   compare this finding to a prior study, the answer is "none".
2. The comparison must be BOUND TO THAT SPECIFIC FINDING. In "left basilar opacity, now
   improved; new small right effusion", "new" binds to the effusion, NOT to the opacity.
3. Respect negation. "no new effusion" is NOT new (answer "none"). "not resolved" /
   "partially resolved" is NOT resolved. A present finding described as having
   "no interval change" IS "persistent".
4. A finding merely mentioned, with no comparison to a prior, is "none" -- this
   includes plain NEGATIVE statements of current state. "there is no pneumothorax",
   "no pleural effusion", "the lungs are clear" simply say the finding is ABSENT NOW;
   they do NOT say it went away, so they are "none", NOT "resolved". Only answer
   "resolved" when the text explicitly signals disappearance relative to a prior
   (e.g. "previously seen X has resolved", "X has cleared", "X is no longer seen").
5. Static size/severity descriptions are not comparisons: "the heart is enlarged" is
   "none" (a state), while "the heart is more enlarged than before" is "persistent".
6. If a finding is not mentioned at all, the answer is "none".

FINDINGS: {findings}

REPORT:
{report}

Output STRICT JSON only -- no prose, no markdown fence -- with exactly these 12 keys:
{{"consolidation":"...","edema":"...","pleural_effusion":"...","pneumonia":"...","pneumothorax":"...","cardiomegaly":"...","enlarged_cardiomediastinum":"...","lung_opacity":"...","lung_lesion":"...","atelectasis":"...","pleural_other":"...","fracture":"..."}}"""


def post(path, payload, timeout=300):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def model_digest(model):
    try:
        d = post("/api/show", {"model": model})
        return (d.get("details") or {}).get("parameter_size", "") + "|" + str(d.get("modified_at", ""))
    except Exception:
        return "unknown"


def parse_json(text):
    """Extract the first JSON object; tolerate fences/prose/<think>. Returns a clean dict or None."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    return {f: (str(d.get(f, "none")).strip().lower()
                if str(d.get(f, "none")).strip().lower() in CLASSES else "none") for f in FIND}


def ask(model, report, retries=2):
    """Returns (transitions dict, ok). ok=False -> parse failure, defaulted to all-none."""
    prompt = PROMPT.format(findings=", ".join(FIND), report=(report or "").strip())
    for attempt in range(retries + 1):
        try:
            r = post("/api/generate", {"model": model, "prompt": prompt, "stream": False,
                                       "think": False, "format": "json",
                                       "options": {"temperature": 0, "top_p": 1, "seed": 0,
                                                   "num_predict": 400}})
            d = parse_json(r.get("response", ""))
            if d:
                return d, True
        except Exception as e:
            if attempt == retries:
                print(f"    ! request failed: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(1.5 * (attempt + 1))
    return {f: "none" for f in FIND}, False


def _gen(path, ids):
    d = json.load(open(path))
    return {i: (d[i][0] if isinstance(d[i], list) else d[i]) for i in ids if i in d}


def load_inputs():
    """Report/generation text for the three systems on the with-prior cohort. GT text and the cohort
    come from the LLM-RG4 annotation (MIMIC-CXR under DUA, not shipped); PriorEdit/sw are shipped."""
    ann = json.load(open(f"{RG}/data/swcmp/annotation.json"))["test"]
    byid = {str(r["id"]): r for r in ann}
    prior = json.load(open(os.path.join(paths.LABELS, "prior_chexpert_labels.json")))
    gt = json.load(open(os.path.join(paths.LABELS, "gt_chexpert_labels.json")))
    ids = sorted(i for i in prior if i in gt and i in byid)
    texts = {"gtself": {i: byid[i].get("finding", "") for i in ids},
             "prioredit": _gen(os.path.join(paths.GENERATIONS, "prioredit.json"), ids),
             "sw": _gen(os.path.join(paths.GENERATIONS, "llmrg4_sw.json"), ids)}
    return ids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["gtself", "prioredit", "sw"])
    ap.add_argument("--model", default="qwen3:30b-a3b")
    ap.add_argument("--n", type=int, default=0, help="0 = all 1337; else a random sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="cache filename suffix (default: n<N>)")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    ids, texts = load_inputs()
    sub = ids
    if a.n and a.n < len(ids):
        sub = sorted(random.Random(a.seed).sample(ids, a.n))
    tag = a.tag or f"n{len(sub)}"
    outp = os.path.join(OUTDIR, f"asserted_{a.system}_{tag}.json")

    cache = json.load(open(outp)) if os.path.exists(outp) else {}
    todo = [i for i in sub if i not in cache]
    print(f"[{a.system}] model={a.model} cohort={len(ids)} sample={len(sub)} "
          f"cached={len(cache)} todo={len(todo)} -> {outp}")

    t0, fails = time.time(), 0
    for k, i in enumerate(todo, 1):
        d, ok = ask(a.model, texts[a.system].get(i, ""))
        fails += not ok
        cache[i] = d
        if k % 10 == 0 or k == len(todo):
            rate = (time.time() - t0) / k
            print(f"  {k}/{len(todo)}  {rate:.2f}s/report  eta {(len(todo)-k)*rate/60:.1f}m  "
                  f"parse_fail={fails}", flush=True)
            json.dump(cache, open(outp, "w"))
    json.dump(cache, open(outp, "w"))

    meta = {"model": a.model, "digest": model_digest(a.model), "temperature": 0, "think": False,
            "format": "json", "seed": a.seed, "n": len(sub), "system": a.system,
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()[:16], "findings": FIND,
            "parse_fail": fails, "sec_per_report": round((time.time() - t0) / max(1, len(todo)), 3),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(meta, open(os.path.join(OUTDIR, f"meta_{a.system}_{tag}.json"), "w"), indent=1)
    print(f"[{a.system}] done: {len(cache)} cached, parse_fail={fails}")


if __name__ == "__main__":
    main()
