#!/usr/bin/env python
"""Run CXRMate (aehrc/cxrmate, longitudinal prior-report-conditioned) on our sw-test studies, to test
whether a DIFFERENT architecture that feeds the prior report into generation also shows the temporal
echo / hallucination. Feeds current frontal image + prior findings/impression (from swcmp last_finding /
last_impression). Output: {id: findings_text}. Then score with temporal_halluc_official / temporal_assessment.

  python run_cxrmate.py --ann <swcmp annotation.json> --out <json> [--with_prior_only]
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import os, re, json, argparse
import torch, transformers
from PIL import Image
from torchvision import transforms

CKPT = "aehrc/cxrmate"
BASE = os.path.join(paths.DATA, "mimic-cxr-jpg/files/")


def clean_section(t):
    """swcmp last_finding/last_impression -> raw section text, or None if empty/'s'."""
    if not t or t.strip() in ("", "s"):
        return None
    t = re.sub(r"^\s*(finding|findings|impression)\s*:\s*", "", t.strip(), flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with_prior_only", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()

    dev = torch.device("cuda")
    os.environ.setdefault("HF_HOME", os.path.join(paths.DATA, "hf-cache"))
    model = transformers.AutoModel.from_pretrained(CKPT, trust_remote_code=True).to(dev).eval()
    tok = transformers.PreTrainedTokenizerFast.from_pretrained(CKPT)
    ip = transformers.AutoFeatureExtractor.from_pretrained(CKPT)
    tf = transforms.Compose([
        transforms.Resize(size=ip.size["shortest_edge"]),
        transforms.CenterCrop(size=[ip.size["shortest_edge"], ip.size["shortest_edge"]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=ip.image_mean, std=ip.image_std),
    ])
    sp_ids = [
        tok.additional_special_tokens_ids[tok.additional_special_tokens.index("[PMT-SEP]")],
        tok.bos_token_id, tok.sep_token_id,
    ]

    recs = json.load(open(a.ann))["test"]
    if a.with_prior_only:
        recs = [r for r in recs if clean_section(r.get("last_finding", "")) or clean_section(r.get("last_impression", ""))]
    print(f"running CXRMate on {len(recs)} studies", flush=True)

    out = {}
    for s in range(0, len(recs), a.batch_size):
        batch = recs[s:s + a.batch_size]
        studies = []
        for r in batch:
            img = Image.open(BASE + r["APPA_imagepath"]).convert("RGB")
            studies.append(tf(img).unsqueeze(0))  # single frontal per study -> [1,3,384,384]
        images = torch.nn.utils.rnn.pad_sequence(studies, batch_first=True, padding_value=0.0).to(dev)
        prev_f = [clean_section(r.get("last_finding", "")) for r in batch]
        prev_i = [clean_section(r.get("last_impression", "")) for r in batch]
        prompt = model.tokenize_prompt(prev_f, prev_i, tok, 256, add_bos_token_id=True)
        with torch.no_grad():
            o = model.generate(
                pixel_values=images,
                decoder_input_ids=prompt["input_ids"],
                special_token_ids=sp_ids,
                bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id, mask_token_id=tok.pad_token_id,
                return_dict_in_generate=True, use_cache=True,
                max_length=256 + prompt["input_ids"].shape[1], num_beams=4,
            )
        seq = o.sequences
        if torch.all(seq[:, 0] == 1):
            seq = seq[:, 1:]
        _, findings, _imp = model.split_and_decode_sections(
            seq, [tok.bos_token_id, tok.sep_token_id, tok.eos_token_id], tok)
        for r, f in zip(batch, findings):
            out[r["id"]] = f
        if (s // a.batch_size) % 10 == 0:
            print(f"  {s + len(batch)}/{len(recs)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"dumped {len(out)} -> {a.out}")


if __name__ == "__main__":
    main()
