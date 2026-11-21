import math
import numpy as np
from pathlib import Path
import tiktoken

text_file = Path(__file__).parent / 'stories.txt'

#settings
embedding_size = 128      # numbers per token
context_length = 64       # how many tokens it can see at once
number_of_heads = 4       # each token asks 4 questions at once
head_size = embedding_size // number_of_heads   # 128 / 4 = 32 numbers per head
number_of_blocks = 6      # block 1 feeds block 2 feeds block 3 ... feeds block 6
batch_size = 16           # how many windows per training step
training_steps = 3000
learning_rate = 3e-3
warmup_steps = 100

# gpt 4o encoding
encoding = tiktoken.get_encoding("o200k_base")

with open(text_file, 'r') as file:
    text = file.read()
tokens = encoding.encode(text)

#we only keep the ids the stories actually use and renumber them 0..4840
used_ids = sorted(set(tokens))
to_small = {big: small for small, big in enumerate(used_ids)}
token_ids = np.array([to_small[t] for t in tokens])
vocab_size = len(used_ids)

#first 90% to train on, last 10% to test on (text it never sees)
cut = int(len(token_ids) * 0.9)
train_ids, test_ids = token_ids[:cut], token_ids[cut:]

random_generator = np.random.default_rng(0)

#all the numbers the model learns go in this one dict
#weights start as small random numbers, biases and shifts start at 0, scales start at 1
params = {}
params["token_embedding_table"] = random_generator.normal(0, 0.02, (vocab_size, embedding_size))
params["position_embedding_table"] = random_generator.normal(0, 0.02, (context_length, embedding_size))
for i in range(number_of_blocks):
    b = f"block{i + 1}."      # block1. block2. ... block6.
    params[b + "norm1_scale"] = np.ones(embedding_size)
    params[b + "norm1_shift"] = np.zeros(embedding_size)
    params[b + "qkv_weights"] = random_generator.normal(0, 0.02, (embedding_size, 3 * embedding_size))
    params[b + "qkv_bias"] = np.zeros(3 * embedding_size)
    params[b + "output_weights"] = random_generator.normal(0, 0.02, (embedding_size, embedding_size))
    params[b + "output_bias"] = np.zeros(embedding_size)
    params[b + "norm2_scale"] = np.ones(embedding_size)
    params[b + "norm2_shift"] = np.zeros(embedding_size)
    params[b + "expand_weights"] = random_generator.normal(0, 0.02, (embedding_size, 4 * embedding_size))
    params[b + "expand_bias"] = np.zeros(4 * embedding_size)
    params[b + "shrink_weights"] = random_generator.normal(0, 0.02, (4 * embedding_size, embedding_size))
    params[b + "shrink_bias"] = np.zeros(embedding_size)
params["final_norm_scale"] = np.ones(embedding_size)
params["final_norm_shift"] = np.zeros(embedding_size)
params["head_weights"] = random_generator.normal(0, 0.02, (embedding_size, vocab_size))
params["head_bias"] = np.zeros(vocab_size)


#we layer norm our vectors, so they have mean 0 and std 1, then scale and shift them
#also returns the stuff backward needs later
def layer_norm(vectors, scale, shift, tiny=1e-5):
    average = vectors.mean(axis=-1, keepdims=True)
    variance = vectors.var(axis=-1, keepdims=True)
    one_over_spread = 1 / np.sqrt(variance + tiny)
    normalized = (vectors - average) * one_over_spread
    return normalized * scale + shift, (normalized, one_over_spread, scale)

#matrix multiply by the weights and add the bias
def linear(vectors, weights, bias):
    return vectors @ weights + bias, (vectors, weights)

#makes every row add up to 1 (subtracting the max first so e^x doesn't blow up)
def softmax(scores):
    exps = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)

#the bend: positives pass, negatives get squashed to about 0
def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


# ---------------- forward: tokens in, guesses out ----------------

#the whole model: token ids (B, T) -> a score for every possible next token (B, T, vocab_size)
#B = how many windows, T = tokens per window
def forward(ids):
    B, T = ids.shape

    #each token's meaning + where it sits
    vectors = params["token_embedding_table"][ids] + params["position_embedding_table"][:T]

    #block 1 -> block 2 -> ... -> block 6. each block takes the vectors the last one made and adds to them
    saved = []
    for i in range(number_of_blocks):
        b = f"block{i + 1}."
        s = {}     # stuff this block's backward will need

        # --- attention: each token grabs info from the tokens before it ---
        normed, s["norm1"] = layer_norm(vectors, params[b + "norm1_scale"], params[b + "norm1_shift"])

        #every token makes a query (what am I looking for), key (what do I have), value (what I pass along)
        qkv, s["qkv"] = linear(normed, params[b + "qkv_weights"], params[b + "qkv_bias"])
        queries, keys, values = np.split(qkv, 3, axis=-1)

        #cut each one into 4 heads of 32: (B, T, 128) -> (B, T, 4, 32) -> (B, 4, T, 32)
        queries = queries.reshape(B, T, number_of_heads, head_size).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, T, number_of_heads, head_size).transpose(0, 2, 1, 3)
        values = values.reshape(B, T, number_of_heads, head_size).transpose(0, 2, 1, 3)

        #score every query against every key: (B, 4, T, T). scaled down so softmax isn't too extreme
        scores = queries @ keys.transpose(0, 1, 3, 2) / np.sqrt(head_size)

        #hide future tokens so it can't cheat by looking at the answer
        future = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(future, -1e9, scores)

        #scores -> weights that add to 1, then add up the values using those weights
        weights = softmax(scores)
        gathered = weights @ values

        #glue the 4 heads back together: (B, 4, T, 32) -> (B, T, 128), then one more matrix
        gathered = gathered.transpose(0, 2, 1, 3).reshape(B, T, embedding_size)
        attended, s["output"] = linear(gathered, params[b + "output_weights"], params[b + "output_bias"])

        #add it on top instead of replacing (keep what it knew, add what it learned)
        vectors = vectors + attended

        # --- mlp: each token thinks on its own. widen to 512, bend, shrink back to 128 ---
        normed, s["norm2"] = layer_norm(vectors, params[b + "norm2_scale"], params[b + "norm2_shift"])
        wide, s["expand"] = linear(normed, params[b + "expand_weights"], params[b + "expand_bias"])
        processed, s["shrink"] = linear(gelu(wide), params[b + "shrink_weights"], params[b + "shrink_bias"])
        vectors = vectors + processed

        s["wide"], s["queries"], s["keys"], s["values"], s["weights"] = wide, queries, keys, values, weights
        saved.append(s)

    #one last norm, then turn each token's 128 numbers into a score for every token in the vocab
    vectors, final_norm = layer_norm(vectors, params["final_norm_scale"], params["final_norm_shift"])
    logits, head = linear(vectors, params["head_weights"], params["head_bias"])
    return logits, (ids, saved, final_norm, head)

#how wrong it was: average of -log(probability it gave the right token)
def get_loss(logits, targets):
    logits, targets = logits.reshape(-1, vocab_size), targets.reshape(-1)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_of_total = np.log(np.exp(shifted).sum(axis=-1))
    right_answer_scores = shifted[np.arange(len(targets)), targets]
    return float(np.mean(log_of_total - right_answer_scores))


# ---------------- backward: work out how to nudge every number ----------------
# same as forward but in reverse: start at the guesses and go back down to the embedding tables

def linear_backward(output_gradient, cache):
    vectors, weights = cache
    flat_vectors = vectors.reshape(-1, vectors.shape[-1])
    flat_gradient = output_gradient.reshape(-1, output_gradient.shape[-1])
    return output_gradient @ weights.T, flat_vectors.T @ flat_gradient, flat_gradient.sum(axis=0)

def layer_norm_backward(output_gradient, cache):
    normalized, one_over_spread, scale = cache
    D = normalized.shape[-1]
    scale_gradient = (output_gradient * normalized).reshape(-1, D).sum(axis=0)
    shift_gradient = output_gradient.reshape(-1, D).sum(axis=0)
    g = output_gradient * scale
    input_gradient = one_over_spread * (g - g.mean(axis=-1, keepdims=True) - normalized * (g * normalized).mean(axis=-1, keepdims=True))
    return input_gradient, scale_gradient, shift_gradient

def gelu_backward(x, output_gradient):
    t = np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
    slope = 0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * 0.7978845608 * (1 + 3 * 0.044715 * x ** 2)
    return output_gradient * slope

def backward(logits, targets, cache):
    ids, saved, final_norm, head = cache
    B, T = ids.shape
    grads = {}

    #start: probabilities minus 1 at the right answer (push the right token up, the rest down)
    flat_targets = targets.reshape(-1)
    gradient = softmax(logits.reshape(-1, vocab_size))
    gradient[np.arange(len(flat_targets)), flat_targets] -= 1
    gradient = (gradient / len(flat_targets)).reshape(logits.shape)

    gradient, grads["head_weights"], grads["head_bias"] = linear_backward(gradient, head)
    gradient, grads["final_norm_scale"], grads["final_norm_shift"] = layer_norm_backward(gradient, final_norm)

    #block 6 -> block 5 -> ... -> block 1
    #at each "vectors + something" the gradient goes down both paths and they add up
    for i in reversed(range(number_of_blocks)):
        b = f"block{i + 1}."
        s = saved[i]

        # --- back through the mlp ---
        g, grads[b + "shrink_weights"], grads[b + "shrink_bias"] = linear_backward(gradient, s["shrink"])
        g = gelu_backward(s["wide"], g)
        g, grads[b + "expand_weights"], grads[b + "expand_bias"] = linear_backward(g, s["expand"])
        g, grads[b + "norm2_scale"], grads[b + "norm2_shift"] = layer_norm_backward(g, s["norm2"])
        gradient = gradient + g

        # --- back through attention ---
        queries, keys, values, weights = s["queries"], s["keys"], s["values"], s["weights"]
        g, grads[b + "output_weights"], grads[b + "output_bias"] = linear_backward(gradient, s["output"])
        g = g.reshape(B, T, number_of_heads, head_size).transpose(0, 2, 1, 3)     # back into heads
        weights_gradient = g @ values.transpose(0, 1, 3, 2)
        values_gradient = weights.transpose(0, 1, 3, 2) @ g
        #back through softmax and the scaling (hidden future spots have weight 0 so they get 0 here)
        scores_gradient = weights * (weights_gradient - (weights_gradient * weights).sum(axis=-1, keepdims=True)) / np.sqrt(head_size)
        queries_gradient = scores_gradient @ keys
        keys_gradient = scores_gradient.transpose(0, 1, 3, 2) @ queries
        #glue heads back and put q, k, v side by side like they came out of the qkv matrix
        g = np.concatenate([
            queries_gradient.transpose(0, 2, 1, 3).reshape(B, T, embedding_size),
            keys_gradient.transpose(0, 2, 1, 3).reshape(B, T, embedding_size),
            values_gradient.transpose(0, 2, 1, 3).reshape(B, T, embedding_size),
        ], axis=-1)
        g, grads[b + "qkv_weights"], grads[b + "qkv_bias"] = linear_backward(g, s["qkv"])
        g, grads[b + "norm1_scale"], grads[b + "norm1_shift"] = layer_norm_backward(g, s["norm1"])
        gradient = gradient + g

    #only rows that got used get a gradient. np.add.at so a token used twice gets both pushes
    grads["token_embedding_table"] = np.zeros_like(params["token_embedding_table"])
    np.add.at(grads["token_embedding_table"], ids, gradient)
    grads["position_embedding_table"] = np.zeros_like(params["position_embedding_table"])
    grads["position_embedding_table"][:T] = gradient.sum(axis=0)
    return grads


# ---------------- training ----------------

#grab random windows. the targets are the same windows shifted by one (the next token at every spot)
def get_batch(ids, batch_random):
    starts = batch_random.integers(0, len(ids) - context_length - 1, size=batch_size)
    inputs = np.stack([ids[s : s + context_length] for s in starts])
    targets = np.stack([ids[s + 1 : s + context_length + 1] for s in starts])
    return inputs, targets

#loss on the test text, same windows every time so before/after is fair
def test_loss():
    test_random = np.random.default_rng(123)
    losses = []
    for _ in range(20):
        inputs, targets = get_batch(test_ids, test_random)
        losses.append(get_loss(forward(inputs)[0], targets))
    return float(np.mean(losses))

#step size ramps up for 100 steps, then slides down to 0 on a cosine curve
def learning_rate_at(step):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (training_steps - warmup_steps)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

print(f"{len(token_ids):,} tokens, {vocab_size:,} different ones")
print(f"{sum(a.size for a in params.values()):,} numbers to learn")
print(f"test loss before training: {test_loss():.3f} (random guessing: {math.log(vocab_size):.3f})")

#adam remembers each number's average gradient (direction) and average squared gradient (size)
average_gradient = {name: np.zeros_like(a) for name, a in params.items()}
average_squared = {name: np.zeros_like(a) for name, a in params.items()}
batch_random = np.random.default_rng(0)

for step in range(training_steps):
    #guess, measure how wrong, work out the gradients
    inputs, targets = get_batch(train_ids, batch_random)
    logits, cache = forward(inputs)
    loss = get_loss(logits, targets)
    grads = backward(logits, targets, cache)

    #if the gradients are huge, shrink them all so one bad batch can't wreck everything
    length = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    if length > 1.0:
        for g in grads.values():
            g *= 1.0 / length

    #adam: nudge every number a little bit downhill
    for name in params:
        average_gradient[name] = 0.9 * average_gradient[name] + 0.1 * grads[name]
        average_squared[name] = 0.95 * average_squared[name] + 0.05 * grads[name] ** 2
        direction = average_gradient[name] / (1 - 0.9 ** (step + 1))
        size = average_squared[name] / (1 - 0.95 ** (step + 1))
        params[name] -= learning_rate_at(step) * direction / (np.sqrt(size) + 1e-8)

    if step == 0 or (step + 1) % 100 == 0:
        print(f"step {step + 1}  loss {loss:.4f}", flush=True)

print(f"test loss after training: {test_loss():.3f}")


# ---------------- writing ----------------

#keep only the 40 best guesses, pick one at random (better guesses more likely), add it, repeat
def generate(prompt, new_tokens=80, temperature=0.8, top_k=40, seed=0):
    write_random = np.random.default_rng(seed)
    ids = [to_small[t] for t in encoding.encode(prompt) if t in to_small]
    for _ in range(new_tokens):
        recent = np.array([ids[-context_length:]])
        scores = forward(recent)[0][0, -1] / temperature
        scores = np.where(scores < np.sort(scores)[-top_k], -1e9, scores)
        probabilities = softmax(scores)
        ids.append(int(write_random.choice(vocab_size, p=probabilities)))
    return encoding.decode([used_ids[i] for i in ids])

for n, prompt in enumerate(["Once upon a time, there was a", "Tom and Sam went to the park.", "The little cat was sad because"]):
    print("\n" + generate(prompt, seed=n))
