# builds lessons/embedding-map.html: a 3d map of the token embeddings the RunPod model learned
#   ../.venv/bin/python make_embedding_map.py   (from the runpod folder, takes about 20 seconds)
import json
import numpy as np
import embedding_map_data as e

d = e.build()
names = list(e.GROUPS) + ["punctuation"]

def fit(xyz):
    # center, then scale so almost every dot sits inside a ball of radius 1 (a few far outliers can poke out)
    xyz = xyz - xyz.mean(axis=0)
    radius = np.percentile(np.linalg.norm(xyz, axis=1), 98)
    return np.round(xyz / radius, 4).ravel().tolist()

data = {
    "labels": d["labels"], "counts": d["counts"],
    "groups": [names.index(g) if g else -1 for g in d["groups"]], "group_names": names,
    "tsne": fit(d["tsne"]), "pca": fit(d["pca"]),
    "neighbors": d["neighbors"].tolist(),
    "sims": [[round(float(d["sims"][i, j]), 2) for j in row] for i, row in enumerate(d["neighbors"])],
}
template = (e.m.here / "embedding_map_template.html").read_text()
html = (template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__COUNT__", f"{len(d['labels']):,}")
        .replace("__EXPLAINED__", f"{sum(d['explained']) * 100:.0f}"))
out = e.m.here.parent / "lessons" / "embedding-map.html"
out.write_text(html)
print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(d['labels'])} tokens)")
