# picks the common tokens, squashes their 256 learned numbers down to 3 (two ways), and finds each one's nearest neighbors.
# used by make_embedding_map.py
import re
from collections import Counter
from pathlib import Path
import numpy as np
import write_story as m

GROUPS = {
    "names": "lily tom tim sue max ben mia anna sam jack lucy sara sarah timmy bob amy joe kate jane emma lila zoe leo molly billy jill ellie mike john spot fluffy mittens bella daisy",
    "animals": "dog cat bird fish bear rabbit bunny duck frog lion mouse cow pig horse elephant monkey owl puppy kitten fox sheep chicken bee butterfly squirrel turtle snake tiger giraffe ant bug deer wolf goat dogs cats birds",
    "colors": "red blue green yellow pink purple orange black white brown gray grey golden gold silver",
    "numbers": "one two three four five six seven eight nine ten twenty hundred first second third 0 1 2 3 4 5 6 7 8 9",
    "family": "mom dad mommy daddy mother father sister brother grandma grandpa baby family aunt uncle parents",
    "feelings": "happy sad angry scared excited sorry proud afraid glad upset worried lonely curious surprised mad nervous brave",
    "food": "cake cookie cookies apple apples candy pizza bread milk soup banana carrot juice sandwich pie cheese honey berries treat",
}
LOOKUP = {w: g for g, words in GROUPS.items() for w in words.split()}

def group_of(text):
    word = text.strip()
    if word and not re.search(r"[A-Za-z0-9]", word):
        return "punctuation"
    g = LOOKUP.get(word.lower())
    if g == "names" and not word[:1].isupper():
        return None
    return g

def build(min_count=4):
    text = (Path(m.here).parent / "stories.txt").read_text()
    counts = Counter(m.encoding.encode_ordinary(text))
    keep = [big for big, c in counts.most_common() if c >= min_count and big in m.to_small]
    rows = np.array([m.to_small[b] for b in keep])
    vectors = m.params["token_embedding_table"][rows].astype(np.float64)
    labels = [m.encoding.decode([b]) for b in keep]

    centered = vectors - vectors.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    pca = centered @ vt[:3].T
    explained = (s[:3] ** 2 / (s ** 2).sum()).tolist()

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3, metric="cosine", perplexity=30, init="pca", random_state=0).fit_transform(vectors)

    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = unit @ unit.T
    np.fill_diagonal(sims, -1)
    neighbors = np.argsort(-sims, axis=1)[:, :8]
    return {"labels": labels, "counts": [counts[b] for b in keep], "groups": [group_of(t) for t in labels],
            "pca": pca, "tsne": tsne, "explained": explained, "neighbors": neighbors, "sims": sims}

if __name__ == "__main__":
    d = build()
    print(len(d["labels"]), "tokens; PCA explains", [round(e, 3) for e in d["explained"]])
    for w in [" dog", "Lily", " red", " happy", " three", " mom", " cake", " ran"]:
        i = d["labels"].index(w)
        print(repr(w), "->", [d["labels"][j] for j in d["neighbors"][i][:6]])
    from collections import Counter as C
    print(C(g for g in d["groups"] if g))
