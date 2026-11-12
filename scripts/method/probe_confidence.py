#!/usr/bin/env python
"""Per-finding pos/neg confidence probe c_f(I) (paper Eq. 1), WITHOUT touching LLM-RG4 code.

Imports the frozen LLM_RG4 model as a library and drives a manual test loop. For each study it wraps the
current image exactly as the sn path does (encode_APPA_img -> zero-pad 8192 -> iit_proj/layernorm ->
prompt_sn -> append BOS), then teacher-force-scores two parallel templates that differ only by negation:
    pos = "There is {finding}."      neg = "There is no {finding}."
confidence[finding] = sum_logP(pos) - sum_logP(neg)   (>0 => model leans positive)

Dumps {study_id: {finding: confidence}} to <savedmodel_path>/result/finding_confidence.json.
Run in the LLM-RG4 environment ($PRIOREDIT_PY_MODEL) with CUDA_VISIBLE_DEVICES set. Arguments mirror
LLM-RG4's own test entry point. The script changes into $PRIOREDIT_LLMRG4 at import, so pass an
absolute --save path.
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import os, sys, json, argparse
import torch

RG = paths.LLMRG4
sys.path.insert(0, RG)
os.chdir(RG)  # model uses relative ./hf and ./save paths

from config.config import parser as rg4_parser          # noqa: E402
from dataset.data_module import DataModule               # noqa: E402
from models.LLM_RG4 import LLM_RG4                        # noqa: E402

# our 14-label -> natural phrase (parallel pos/neg templates differ only by "no")
FINDING_PHRASES = {
    "enlarged_cardiomediastinum": "an enlarged cardiomediastinum",
    "cardiomegaly": "cardiomegaly",
    "lung_opacity": "a lung opacity",
    "lung_lesion": "a pulmonary nodule",
    "edema": "pulmonary edema",
    "consolidation": "consolidation",
    "pneumonia": "pneumonia",
    "atelectasis": "atelectasis",
    "pneumothorax": "pneumothorax",
    "pleural_effusion": "pleural effusion",
    "pleural_other": "pleural thickening",
    "fracture": "a fracture",
    "support_devices": "a support device",
}


def wrap_sn(model, samples):
    """Replicate the sn image+prompt wrapping, returning prefix embeds/mask (with trailing BOS)."""
    img_embeds1, atts_img1, _ = model.encode_APPA_img(samples["image"])
    zero_padding = torch.zeros(img_embeds1.size(0), img_embeds1.size(1), 8192, device=img_embeds1.device)
    new_embeds = torch.cat([img_embeds1, zero_padding], dim=2)
    new_embeds = model.iit_proj(new_embeds)
    new_embeds = model.iit_layernorm(new_embeds)
    img_embeds, atts_img = model.prompt_sn(new_embeds, atts_img1, samples)
    bsz = img_embeds.size(0)
    bos = torch.ones([bsz, 1], dtype=atts_img.dtype, device=atts_img.device) * model.llama_tokenizer.bos_token_id
    bos_embeds = model.embed_tokens(bos)
    atts_bos = atts_img[:, -1:]
    prefix = torch.cat([img_embeds, bos_embeds], dim=1)
    mask = torch.cat([atts_img, atts_bos], dim=1)
    return prefix, mask


def score_text(model, prefix, mask, text):
    """Sum logP the model assigns to `text` as a continuation of prefix. Returns [B]."""
    bsz = prefix.size(0)
    model.llama_tokenizer.padding_side = "right"
    tok = model.llama_tokenizer([text] * bsz, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(prefix.device)
    cand = model.embed_tokens(tok.input_ids)
    inp = torch.cat([prefix, cand], dim=1)
    amask = torch.cat([mask, tok.attention_mask], dim=1)
    out = model.llama_model(inputs_embeds=inp, attention_mask=amask, return_dict=True)
    logp = torch.log_softmax(out.logits.float(), dim=-1)
    Lp, cl = prefix.size(1), tok.input_ids.size(1)
    pos = torch.arange(cl, device=inp.device) + Lp - 1     # logit at Lp+i-1 predicts cand token i
    sel = logp[:, pos, :].gather(-1, tok.input_ids.unsqueeze(-1)).squeeze(-1)
    return (sel * tok.attention_mask.float()).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--delta", default=f"{RG}/save/llmrg4_mimiccxr_findingonly.pth")
    known, _ = ap.parse_known_args()

    argv = [
        "--test", "--dataset", "mimic_cxr",
        "--sn_annotation", known.ann, "--sw_annotation", known.ann,
        "--mn_annotation", known.ann, "--mw_annotation", known.ann,
        "--base_dir", os.path.join(paths.DATA, "mimic-cxr-jpg/files/"),
        "--vicuna_model", "./hf/vicuna-7b-v1.5", "--rad_dino_path", "./hf/rad-dino",
        "--cxr_bert_path", "./hf/BiomedVLP-CXR-BERT-specialized",
        "--chexbert_path", "./hf/chexbert.pth", "--bert_path", "./hf/bert-base-uncased",
        "--delta_file", known.delta,
        "--savedmodel_path", known.save, "--test_mode", "sn", "--stage_class", "2",
        "--llm_use_lora", "True", "--llm_r", "32", "--llm_alpha", "64", "--lora_dropout", "0.1",
        "--visual_token_number", "128", "--loss_mode", "sentence", "--sentence_ratio", "0.75",
        "--test_batch_size", str(known.batch_size), "--num_workers", "8",
        "--devices", "1", "--precision", "bf16-mixed",
    ]
    args = rg4_parser.parse_args(argv)

    dm = DataModule(args); dm.setup("test")
    loader = dm.test_dataloader()
    model = LLM_RG4(args).cuda().eval()

    out = {}
    n = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for samples in loader:
            samples["image"] = samples["image"].cuda()
            prefix, mask = wrap_sn(model, samples)
            conf = {}
            for k, ph in FINDING_PHRASES.items():
                posl = score_text(model, prefix, mask, f"There is {ph}.")
                negl = score_text(model, prefix, mask, f"There is no {ph}.")
                conf[k] = (posl - negl).detach().float().cpu().tolist()
            for j, sid in enumerate(samples["id"]):
                out[sid] = {k: conf[k][j] for k in FINDING_PHRASES}
            n += len(samples["id"])
            if n % 120 == 0:
                print(f"  {n} studies scored", flush=True)

    rf = os.path.join(known.save, "result"); os.makedirs(rf, exist_ok=True)
    dst = os.path.join(rf, "finding_confidence.json")
    json.dump(out, open(dst, "w"))
    print(f"dumped confidence for {len(out)} studies -> {dst}")


if __name__ == "__main__":
    main()
