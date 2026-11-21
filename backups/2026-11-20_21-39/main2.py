# main2.py: Claude's working copy. Asher reads this and types what they want into main.py themselves.
# main3.py is this same file with the comments stripped out. Run it with:  .venv/bin/python main2.py
from pathlib import Path
import tiktoken
import numpy as np

# [1] TEXT
# Path(__file__) is this file, so .parent is the gpt-one-file folder. Works no matter where you run it from.
text_file = Path(__file__).parent / 'stories.txt'

# [2] TOKENS
# gpt 4o encoding
encoding = tiktoken.get_encoding("o200k_base")

def tokenize(text_file_input=text_file):
    with open(text_file_input, 'r') as file:
        text = file.read()
    return encoding.encode(text)

tokens = tokenize(text_file)
token_readable = [encoding.decode([t]) for t in tokens]

print(len(tokens))                                   # how many tokens the whole file becomes
print(token_readable[:20])                           # decode each id back to its text, to see the pieces

# Renumber: tiktoken ids go up to ~200,000 but only 4,841 different ones appear in stories.txt.
# Give each used id a small number 0..4840 so the token table below only needs 4,841 rows.
used_tiktoken_ids = sorted(set(tokens))              # the different tiktoken ids that appear, smallest first
tiktoken_to_our_id = {tiktoken_id: our_id for our_id, tiktoken_id in enumerate(used_tiktoken_ids)}
token_ids = np.array([tiktoken_to_our_id[t] for t in tokens])   # the whole file, in our ids 0..4840
vocab_size = len(used_tiktoken_ids)                  # 4841: how many different tokens the model knows
# to go back later: used_tiktoken_ids[our_id] is the tiktoken id, then encoding.decode([...]) gives the text

# [3] POSITION + TOKEN EMBEDDING
# The model can't do math on an id like 1058. Step 3 swaps each id for a list of 128 numbers (a vector).
#
# Two tables, both just grids of numbers:
#
#   token_embedding_table    (4841, 128)   one row per token in the vocabulary
#       Row 1058 is the 128 numbers for ' time'. These numbers are the token's MEANING. They start random,
#       and training nudges them until tokens used in similar ways (' cat', ' dog') end up with similar rows.
#
#   position_embedding_table (64, 128)     one row per slot in the window: slot 0, slot 1, ... slot 63
#       Row 5 says "I'm the 6th token". Without this, "Tom saw Sam" and "Sam saw Tom" would be the same
#       bag of vectors, because the token rows don't change with where the word is.
#
# For every token in the window:   its vector = token row (what it is) + position row (where it is)
#
# Why start random, and why 0.02: rng.normal(0, 0.02, shape) fills the grid with numbers clustered around 0,
# most between -0.04 and 0.04. Small means no token starts out "louder" than another, and random means no two
# rows start out identical (if they did, training would treat them the same forever).
embedding_size = 128     # how many numbers describe each token (gpt.txt calls this d)
context_length = 64      # how many tokens the model reads at once (gpt.txt calls this max_T)

random_generator = np.random.default_rng(0)          # seeded with 0, so every run gets the same "random" numbers
token_embedding_table = random_generator.normal(0, 0.02, (vocab_size, embedding_size))         # (4841, 128)
position_embedding_table = random_generator.normal(0, 0.02, (context_length, embedding_size))  # (64, 128)

window = token_ids[:context_length]                  # the first 64 token ids of the file: 'Once', ' upon', ...

# token_embedding_table[window] grabs one row per id in window, all at once -> (64, 128).
# position_embedding_table[:64] is rows 0..63 -> (64, 128). Same shape, so they add number by number.
# Row 0 of the result = the row for 'Once' + the row for position 0, and so on down the window.
vectors = token_embedding_table[window] + position_embedding_table[:len(window)]
print(vocab_size, vectors.shape)                     # 4841 (64, 128): 64 tokens, 128 numbers each

# [4] NORMALIZE (LayerNorm)
# Every step after this does math on these vectors, and as the numbers flow through the model some vectors
# drift big and others drift tiny. Big numbers make the later steps swing wildly and training gets unstable.
# LayerNorm fixes the scale of each token's vector on its own, before every big step:
#   1. subtract the average   so the 128 numbers are centered on 0
#   2. divide by the spread   so they spread out by about 1 (the spread is the standard deviation)
#   3. scale and shift        multiply by a learned scale and add a learned shift
#
# Example with 4 numbers instead of 128:
#   [1, 2, 3, 4]   average 2.5           -> [-1.5, -0.5, 0.5, 1.5]
#                  spread 1.118          -> [-1.342, -0.447, 0.447, 1.342]
#
# Step 3 exists because forcing everything to exactly "average 0, spread 1" might throw away something useful.
# The scale starts at all 1s and the shift at all 0s, so at first they change nothing. Training can then
# stretch or move any of the 128 numbers if that helps.
layer_norm_scale = np.ones(embedding_size)           # (128,) gpt.txt calls this g
layer_norm_shift = np.zeros(embedding_size)          # (128,) gpt.txt calls this b

def layer_norm(vectors, scale, shift, tiny=1e-5):
    # axis=-1 means "across the last dimension": each token's 128 numbers, separately for every token.
    # keepdims=True keeps the result as (64, 1) instead of (64,), so it subtracts row by row from (64, 128).
    average = vectors.mean(axis=-1, keepdims=True)   # (64, 1): each token's average
    variance = vectors.var(axis=-1, keepdims=True)   # (64, 1): each token's spread, squared
    spread = np.sqrt(variance + tiny)                # tiny stops a divide-by-zero if all 128 numbers are equal
    normalized = (vectors - average) / spread        # (64, 128): every row now has average 0, spread 1
    return normalized * scale + shift                # (64, 128): the learned stretch and move

normalized_vectors = layer_norm(vectors, layer_norm_scale, layer_norm_shift)

# Check it worked on the first token ('Once'): before, its numbers are tiny (spread ~0.03); after, spread ~1.
# It prints 0.9935, not exactly 1, because of tiny: the variance here is only about 0.00076, so adding
# 0.00001 makes the spread about 0.7% bigger. Once training makes the numbers bigger, it rounds to 1.
print("before:", round(vectors[0].mean(), 4), round(vectors[0].std(), 4))
print("after: ", round(normalized_vectors[0].mean(), 4), round(normalized_vectors[0].std(), 4))
print(layer_norm(np.array([[1.0, 2.0, 3.0, 4.0]]), 1, 0).round(3))   # the example above: [-1.342 -0.447 0.447 1.342]

# [5] SEARCH [Q] · MATCH [K] · VALUE [V]
# This is the start of attention: the one place where tokens look at each other and share information.
# Each token makes three new vectors out of its own (normalized) vector:
#   query   "what am I looking for?"        e.g. ' sad' might look for who is sad
#   key     "what do I have to offer?"      e.g. ' Tom' might offer "I'm a person, I can be sad"
#   value   "what I hand over if I'm picked"  the actual information that gets passed along
# Steps 6-10 match every query against every key, and the tokens that match well pass their values along.
# Right now the numbers are random, so nothing means anything yet. Training is what teaches the model what
# to look for and what to offer.
#
# How a vector becomes a new vector: a matrix multiply ("linear layer").  output = vectors @ weights + bias
#   vectors (64, 128) @ weights (128, n) -> (64, n)
#   Each of the n output numbers is a weighted sum of the token's 128 input numbers, with its own 128 weights
#   (one column of the weights matrix). The bias is one extra number added to each output, starting at 0.
#
# The query, key and value each need 128 numbers, so instead of three matrices of (128, 128) we use one of
# (128, 384) and cut the result into three. It's the same math, done in one multiply.
def linear(vectors, weights, bias):
    return vectors @ weights + bias                  # (64, 128) @ (128, n) + (n,) -> (64, n)

qkv_weights = random_generator.normal(0, 0.02, (embedding_size, 3 * embedding_size))   # (128, 384), small and random
qkv_bias = np.zeros(3 * embedding_size)                                                 # (384,), starts at 0

queries_keys_values = linear(normalized_vectors, qkv_weights, qkv_bias)   # (64, 384): every token's q, k, v side by side
queries, keys, values = np.split(queries_keys_values, 3, axis=-1)       # cut into three, each (64, 128)

# Heads: instead of every token asking ONE question with 128 numbers, it asks 4 smaller questions with 32
# numbers each, all at the same time. One head might learn to look for "who", another for "the last
# verb", and so on. We cut each token's 128 numbers into 4 slices of 32: numbers 0-31 are head 0, 32-63 are
# head 1, etc.
#   reshape   (64, 128) -> (64, 4, 32)   each token's 128 numbers become 4 rows of 32
#   transpose (64, 4, 32) -> (4, 64, 32)  group by head first: head 0's 64 tokens, then head 1's, ...
# After this, each head is its own (64, 32) grid and the next steps run all 4 heads at once.
number_of_heads = 4                                  # gpt.txt: n_heads
head_size = embedding_size // number_of_heads        # 128 / 4 = 32 numbers per head (gpt.txt: hs)

def split_into_heads(vectors):
    number_of_tokens = vectors.shape[0]                                          # 64
    split = vectors.reshape(number_of_tokens, number_of_heads, head_size)        # (64, 4, 32)
    return split.transpose(1, 0, 2)                                              # (4, 64, 32)

queries = split_into_heads(queries)                  # (4, 64, 32)
keys = split_into_heads(keys)                        # (4, 64, 32)
values = split_into_heads(values)                    # (4, 64, 32)

print(queries_keys_values.shape, queries.shape, keys.shape, values.shape)   # (64, 384) (4, 64, 32) x3
# Check the cut: head 1's query for token 0 should be numbers 32-63 of token 0's full 128-number query.
full_queries = np.split(queries_keys_values, 3, axis=-1)[0]                  # (64, 128), before splitting into heads
print(np.array_equal(queries[1, 0], full_queries[0, 32:64]))                 # True
