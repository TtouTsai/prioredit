#!/usr/bin/env python
"""LLM-editor ablation (paper App. F / §8): is decoupling, or just 'rewrite with a model', what helps?

Uses the SAME backbone as PriorEdit's base (LLM-RG4's LoRA Vicuna-7B) as a pure TEXT rewriter: feed the
image-only sn draft plus the prior report text and instruct it to add temporal comparison sentences,
then generate. If a generative editor with the same model still copies prior state forward (hallucinates
temporal change) at sw-level change precision, then the auditable rule editor's grounding -- not model
capacity -- is the active ingredient. In the paper this reaches 35.6% hallucination / 13% change
precision on 300 studies, versus the rule editor's 4.1% / 46% on the same subset.

Generates the reports only; score afterwards with scripts/metric/temporal_assessment.py (change
precision) and scripts/metric/temporal_halluc_official.py (hallucination) on the produced ids.

Imports the frozen LLM_RG4 model as a library and reads only its llama_model / tokenizer; touches no
paper code. Report text and images are MIMIC-CXR under a PhysioNet DUA and are not redistributed here.
Env: the LLM-RG4 inference env; one GPU. See docs/ENVIRONMENTS.md.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots from PRIOREDIT_* env vars)
import os, sys, json, argparse
import torch

RG = paths.LLMRG4
sys.path.insert(0, RG)
os.chdir(RG)  # model uses relative ./hf paths
from config.config import parser as rg4_parser  # noqa: E402
from models.LLM_RG4 import LLM_RG4              # noqa: E402

VICUNA_SYS = ("A chat between a curious user and an artificial intelligence assistant. "
              "The assistant gives helpful, detailed answers to the user's questions.")


def clean(s):
    s = (s or "").strip()
    return "" if s.lower() == "s" else s.replace("finding :", "").replace("finding:", "").strip()


def build_prompt(prior, draft):
    instr = ("Here is a draft chest X-ray FINDINGS report for the current study, and the FINDINGS from "
             "the patient's previous study. Revise the current draft so that it explicitly notes temporal "
             "change versus the prior (use words like 'new', 'resolved', 'unchanged', 'improved', "
             "'worsened') wherever the comparison is warranted, while keeping every finding already in the "
             "draft and not inventing findings. Output only the revised FINDINGS.\n\n"
             f"Previous study FINDINGS: {prior}\n\nCurrent draft FINDINGS: {draft}")
    return f"{VICUNA_SYS} USER: {instr} ASSISTANT:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--drafts", default=f"{RG}/save/swcmp_sn_off/result/test_result.json")
    ap.add_argument("--ann", default=f"{RG}/data/swcmp/annotation.json")
    ap.add_argument("--prior_labels", default=os.path.join(paths.LABELS, "prior_chexpert_labels.json"),
                    help="restrict to studies with a real prior (its keys)")
    ap.add_argument("--delta", default=f"{RG}/save/llmrg4_mimiccxr_findingonly.pth")
    ap.add_argument("--images", default=os.environ.get(
        "PRIOREDIT_MIMIC_JPG", os.path.join(paths.DATA, "mimic-cxr-jpg", "files") + "/"))
    ap.add_argument("--out", default=os.path.join(paths.OUT, "llm_editor", "llm_editor_reports.json"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    argv = ["--test", "--dataset", "mimic_cxr",
            "--sn_annotation", a.ann, "--sw_annotation", a.ann,
            "--mn_annotation", a.ann, "--mw_annotation", a.ann,
            "--base_dir", a.images,
            "--vicuna_model", "./hf/vicuna-7b-v1.5", "--rad_dino_path", "./hf/rad-dino",
            "--cxr_bert_path", "./hf/BiomedVLP-CXR-BERT-specialized", "--chexbert_path", "./hf/chexbert.pth",
            "--bert_path", "./hf/bert-base-uncased", "--delta_file", a.delta,
            "--savedmodel_path", os.path.dirname(a.out),
            "--test_mode", "sn", "--stage_class", "2", "--llm_use_lora", "True", "--llm_r", "32",
            "--llm_alpha", "64", "--lora_dropout", "0.1", "--visual_token_number", "128",
            "--loss_mode", "sentence", "--sentence_ratio", "0.75",
            "--test_batch_size", str(a.batch_size), "--num_workers", "4", "--devices", "1",
            "--precision", "bf16-mixed"]
    model = LLM_RG4(rg4_parser.parse_args(argv)).cuda().eval()
    tok = model.llama_tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    drafts = json.load(open(a.drafts))
    prior_keys = set(json.load(open(a.prior_labels)).keys())
    ann = {str(t["id"]): t for t in json.load(open(a.ann))["test"]}
    ids = [i for i in drafts if i in prior_keys and clean(ann.get(i, {}).get("last_finding", ""))][:a.n]
    print(f"LLM-editor rewrite on {len(ids)} studies with prior", flush=True)

    out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(ids), a.batch_size):
            chunk = ids[s:s + a.batch_size]
            prompts = [build_prompt(clean(ann[i].get("last_finding", "")),
                                    drafts[i][0] if isinstance(drafts[i], list) else drafts[i])
                       for i in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
            gen = model.llama_model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                             max_new_tokens=200, do_sample=False, num_beams=1,
                                             pad_token_id=tok.pad_token_id)
            for i, t in zip(chunk, tok.batch_decode(gen[:, enc.input_ids.size(1):], skip_special_tokens=True)):
                out[i] = t.strip()
            print(f"  {min(s + a.batch_size, len(ids))}/{len(ids)}", flush=True)

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"saved {len(out)} rewritten reports -> {a.out}")


if __name__ == "__main__":
    main()
