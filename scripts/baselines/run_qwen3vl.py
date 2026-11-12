#!/usr/bin/env python
"""Qwen3-VL-2B image-only base (paper App. F "Is a clean base necessary?", Table 7).

Generates the FINDINGS section from the current frontal image alone with the released
dmusingu/Qwen3-VL-2B-RRG-SFT checkpoint (Qwen3-VL-2B fine-tuned for report generation), on the
1337 with-prior test studies (the studies with prior labels in data/labels/). Greedy decoding,
max_new_tokens=180, one fixed instruction. Output is {id: "finding : <text>"}, the same
format as the other generation files, and is resumable (re-running skips cached ids).

  python run_qwen3vl.py --ann $PRIOREDIT_LLMRG4/data/swcmp/annotation.json \
      --out out/qwen3vl_imageonly.json        # shipped as data/generations/qwen3vl_imageonly.json

Then score it like any other generation file:
  TCA (Tier A):  temporal_assessment.py --preds Qwen3-VL:<out>       -> 0 asserted change events
  editor:        rule_editor.py --drafts <out> --ann ... --inplace --high_conf --keep_unchanged
                                                                     -> 21% change precision
  presence/RaTE: make_eval_dir.py + paper_chexbert_eval.py / radeval_score.py   -> Table 7

Env: transformers with Qwen3-VL support (>= 4.57), torch, one GPU (bfloat16, about 5 GB).
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import os, sys, json, time, argparse

BASE = os.path.join(paths.DATA, "mimic-cxr-jpg/files/")
MODEL_ID = "dmusingu/Qwen3-VL-2B-RRG-SFT"
INSTR = ("You are a radiologist. Write the FINDINGS section of the chest X-ray report for this image. "
         "Describe only what is visible in this single image. Use standard radiology phrasing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="LLM-RG4 swcmp annotation.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N studies only (smoke test)")
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", os.path.join(paths.DATA, "hf-cache"))
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    prior = json.load(open(os.path.join(paths.LABELS, "prior_chexpert_labels.json")))
    gt = json.load(open(os.path.join(paths.LABELS, "gt_chexpert_labels.json")))
    ann = json.load(open(a.ann))["test"]
    cohort = sorted((r for r in ann if str(r["id"]) in prior and str(r["id"]) in gt
                     and r.get("APPA_imagepath")), key=lambda r: str(r["id"]))
    if a.limit:
        cohort = cohort[:a.limit]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cache = json.load(open(a.out)) if os.path.exists(a.out) else {}
    todo = [r for r in cohort if str(r["id"]) not in cache]
    print(f"cohort={len(cohort)} cached={len(cache)} todo={len(todo)} -> {a.out}", flush=True)

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        img = BASE + r["APPA_imagepath"]
        try:
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": INSTR}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=[Image.open(img).convert("RGB")],
                       return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=180, do_sample=False)
            gen = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        except Exception as e:  # keep going; an empty draft is scored as silent
            gen = ""
            print(f"  ! {r['id']}: {type(e).__name__}", file=sys.stderr)
        cache[str(r["id"])] = "finding : " + gen
        if k % 25 == 0 or k == len(todo):
            rate = (time.time() - t0) / k
            print(f"  {k}/{len(todo)} {rate:.1f}s/it eta {(len(todo) - k) * rate / 60:.1f}m", flush=True)
            json.dump(cache, open(a.out, "w"))
    json.dump(cache, open(a.out, "w"))
    print(f"done {len(cache)}")


if __name__ == "__main__":
    main()
