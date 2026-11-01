#!/usr/bin/env python
"""Fine-tune CXR-BERT as a 3-class temporal direction classifier (improving/stable/worsening).

Input pair: A = "finding : {fd} . prior : {prior_ctx}"  B = "current : {current_ctx}"  -> CLS -> Linear(3).
Silver labels from build_direction_trainset.py; class-weighted CE handles the worsening-heavy prior. The
masked half of the data forces the model to read severity, not comparison words (deployment drafts have
none). Saves best-macro-F1 checkpoint.

Run (see docs/ENVIRONMENTS.md): python train_direction_verifier.py --epochs 4
"""
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_scripts = _here
while _scripts != "/" and not _os.path.exists(_os.path.join(_scripts, "paths.py")):
    _scripts = _os.path.dirname(_scripts)
_sys.path.insert(0, _scripts)
import paths  # noqa: E402  (roots come from PRIOREDIT_* env vars; see scripts/paths.py)
import json, argparse, os, random
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

CXR = os.path.join(paths.LLMRG4, "hf/BiomedVLP-CXR-BERT-specialized")
DDIR = os.path.join(paths.OUT, "direction")
OUT = os.path.join(paths.OUT, "models/direction_verifier")
LABELS = ["improving", "stable", "worsening"]
L2I = {l: i for i, l in enumerate(LABELS)}


def fmt(r):
    a = f"finding : {r['finding'].replace('_', ' ')} . prior : {r['prior_ctx'] or 'not mentioned'}"
    b = f"current : {r['current_ctx'] or 'not mentioned'}"
    return a, b


class DS(Dataset):
    def __init__(self, path, tok, maxlen=192):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok; self.maxlen = maxlen

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]; a, b = fmt(r)
        enc = self.tok(a, b, truncation=True, max_length=self.maxlen, padding="max_length",
                       return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, L2I[r["label"]]


class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(CXR, trust_remote_code=True)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(768, 3)

    def forward(self, **x):
        h = self.enc(**x).last_hidden_state[:, 0]
        return self.head(self.drop(h))


def evaluate(model, dl, dev):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for enc, y in dl:
            enc = {k: v.to(dev) for k, v in enc.items()}
            logit = model(**enc)
            ps += logit.argmax(1).cpu().tolist(); ys += y.tolist()
    ys, ps = np.array(ys), np.array(ps)
    f1s = []
    for c in range(3):
        tp = ((ps == c) & (ys == c)).sum(); fp = ((ps == c) & (ys != c)).sum(); fn = ((ps != c) & (ys == c)).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1s.append(2 * p * r / (p + r + 1e-9))
    return (ys == ps).mean(), np.mean(f1s), f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--maxlen", type=int, default=192)
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    dev = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(CXR, trust_remote_code=True)
    tr = DS(f"{DDIR}/train.jsonl", tok, a.maxlen); va = DS(f"{DDIR}/val.jsonl", tok, a.maxlen)
    # class weights (inverse freq on train)
    cnt = np.bincount([L2I[r["label"]] for r in tr.rows], minlength=3).astype(float)
    w = (cnt.sum() / (3 * cnt)); print("class counts", cnt, "weights", w.round(3))
    dltr = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=4, drop_last=True)
    dlva = DataLoader(va, batch_size=64, num_workers=4)
    model = Clf().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(dltr) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(steps * 0.06), steps)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=dev))
    os.makedirs(OUT, exist_ok=True)
    best = -1
    for ep in range(a.epochs):
        model.train(); tot = 0
        for i, (enc, y) in enumerate(dltr):
            enc = {k: v.to(dev) for k, v in enc.items()}; y = y.to(dev)
            logit = model(**enc); loss = lossf(logit, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += loss.item()
            if i % 200 == 0:
                print(f"ep{ep} step{i}/{len(dltr)} loss {loss.item():.3f}", flush=True)
        acc, mf1, f1s = evaluate(model, dlva, dev)
        print(f"== ep{ep} val acc {acc:.3f} macroF1 {mf1:.3f} per-class {[round(x,3) for x in f1s]}", flush=True)
        if mf1 > best:
            best = mf1
            torch.save(model.state_dict(), f"{OUT}/best.pt")
            print(f"   saved best -> {OUT}/best.pt (macroF1 {mf1:.3f})", flush=True)
    print(f"done. best val macroF1 {best:.3f}")


if __name__ == "__main__":
    main()
