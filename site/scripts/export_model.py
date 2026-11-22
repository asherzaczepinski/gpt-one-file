# turns runpod/results/out/best_model.pkl into files a browser can load:
#   public/model/weights.bin     every learned number, as float16 (half the size of float32, same stories)
#   public/model/manifest.json   settings, where each array starts in weights.bin, and the token list
#   scripts/fixture.json         the numpy model's answers for a test prompt, so the javascript version can be checked
#   run:  ../.venv/bin/python scripts/export_model.py   (from the site folder)
import json
import pickle
from pathlib import Path
import numpy as np

site = Path(__file__).resolve().parent.parent
project = site.parent
saved = pickle.load(open(project / "runpod" / "results" / "out" / "best_model.pkl", "rb"))
params = saved["params"]
settings = saved["settings"]

out = site / "public" / "model"
out.mkdir(parents=True, exist_ok=True)

tensors, chunks, offset = [], [], 0
for name, array in params.items():
    half = np.asarray(array, dtype=np.float16)
    tensors.append({"name": name, "shape": list(half.shape), "offset": offset})
    chunks.append(half.ravel())
    offset += half.size
np.concatenate(chunks).tofile(out / "weights.bin")
manifest = {"settings": settings, "vocab_size": len(saved["used_ids"]), "total": offset,
            "tensors": tensors, "used_ids": saved["used_ids"]}
(out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
print(f"{offset:,} numbers -> weights.bin {(out / 'weights.bin').stat().st_size / 1e6:.1f} MB")

# the reference answers, from the exact same forward() the playground uses
import sys
sys.path.insert(0, str(project / "runpod"))
import write_story as model

def last_logits(p, ids):
    keep = model.params
    model.params = p
    logits = model.forward(np.array([ids]))[0][0, -1].astype(np.float64)
    model.params = keep
    return logits

prompt = "Once upon a time, there was a little dog named Max. Max liked to"
ids = [model.to_small[t] for t in model.encoding.encode(prompt) if t in model.to_small]
full = last_logits(params, ids)
half = last_logits({k: v.astype(np.float16).astype(np.float32) for k, v in params.items()}, ids)

def int8(a):
    if a.ndim < 2:
        return a
    scale = np.abs(a).max(axis=0, keepdims=True) / 127 + 1e-12
    return (np.round(a / scale).clip(-127, 127) * scale).astype(np.float32)
small = last_logits({k: int8(v) for k, v in params.items()}, ids)

def compare(name, other):
    p, q = np.exp(full - full.max()), np.exp(other - other.max())
    p, q = p / p.sum(), q / q.sum()
    print(f"{name}: biggest logit change {np.abs(full - other).max():.4f}, same top token: {full.argmax() == other.argmax()}, "
          f"top-10 overlap {len(set(np.argsort(full)[-10:]) & set(np.argsort(other)[-10:]))}/10, KL {float((p * np.log(p / q)).sum()):.6f}")
compare("float16", half)
compare("int8   ", small)

top = np.argsort(full)[::-1][:20]
fixture = {"prompt": prompt, "ids": ids, "top_ids": [int(i) for i in top], "top_logits": [float(full[i]) for i in top]}
(site / "scripts" / "fixture.json").write_text(json.dumps(fixture))
print("fixture: first prompt ids", ids[:6], "... top next tokens:",
      [model.encoding.decode([model.used_ids[i]]) for i in top[:5]])
