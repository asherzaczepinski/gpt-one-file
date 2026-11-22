# write stories on the Mac with the model trained on RunPod (no GPU needed, plain numpy)
#   .venv/bin/python runpod/write_story.py                          # the 4 usual prompts
#   .venv/bin/python runpod/write_story.py "Once upon a time, a dog"  # your own prompt
# play.py imports this file too, so the playground uses the exact same model code
import sys
import math
import pickle
from pathlib import Path
import numpy as np
import tiktoken

here = Path(__file__).parent
saved = pickle.load(open(here / "results" / "out" / "best_model.pkl", "rb"))

# the names forward() expects, same as in train_tinystories.py
xp = np
on_gpu = False
dtype = np.float32
scatter_add = np.add.at
def to_cpu(a):
    return a
params = saved["params"]
used_ids = saved["used_ids"]
to_small = {big: small for small, big in enumerate(used_ids)}
vocab_size = len(used_ids)
embedding_size = saved["settings"]["embedding_size"]
context_length = saved["settings"]["context_length"]
number_of_heads = saved["settings"]["number_of_heads"]
number_of_blocks = saved["settings"]["number_of_blocks"]
head_size = embedding_size // number_of_heads
encoding = tiktoken.get_encoding("o200k_base")

# borrow layer_norm, linear, softmax, gelu and forward straight from the trainer, so it's the exact same math
source = (here / "train_tinystories.py").read_text().splitlines()
start = next(i for i, line in enumerate(source) if line.startswith("def layer_norm"))
end = next(i for i, line in enumerate(source) if line.startswith("def get_loss"))
exec("\n".join(source[start:end]))

def next_token_probabilities(ids, temperature=0.8, top_k=40):
    # the model's guess for the token after ids: logits for the last position -> temperature -> keep top_k -> softmax
    recent = np.array([ids[-context_length:]])
    scores = forward(recent)[0][0, -1].astype(np.float64) / temperature
    scores = np.where(scores < np.sort(scores)[-top_k], -1e9, scores)
    probabilities = np.exp(scores - scores.max())
    return probabilities / probabilities.sum()

def generate(prompt, new_tokens=150, temperature=0.8, top_k=40, seed=0):
    write_generator = np.random.default_rng(seed)
    ids = [to_small[t] for t in encoding.encode_ordinary(prompt) if t in to_small]
    for _ in range(new_tokens):
        probabilities = next_token_probabilities(ids, temperature, top_k)
        ids.append(int(write_generator.choice(vocab_size, p=probabilities)))
    return encoding.decode([used_ids[i] for i in ids])

if __name__ == "__main__":
    prompts = sys.argv[1:] or ["Once upon a time, there was a", "Tom and Sam went to the park.",
                               "The little cat was sad because", "Lily found a shiny key in the garden."]
    for prompt in prompts:
        print(generate(prompt))
        print("-" * 60)
