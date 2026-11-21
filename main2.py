# main2.py: Claude's working copy. Asher reads this and types what they want into ../main.py themselves.
# Run it with:  .venv/bin/python for-claude/main2.py   (from the gpt-one-file folder)
from pathlib import Path
import tiktoken
import numpy as np

# [1] TEXT
# Path(__file__) is this file, so .parent.parent is the gpt-one-file folder. Works no matter where you run it from.
text_file = Path(__file__).parent.parent / 'stories.txt'

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

# [3] POSITION + TOKEN EMBEDDING
# Renumber: tiktoken ids go up to ~200,000 but only 4,841 different ones appear in stories.txt.
# Give each used id a small number 0..4840 so the token table only needs 4,841 rows.
used_ids = sorted(set(tokens))                                     # the different ids that appear, smallest first
to_small = {big: small for small, big in enumerate(used_ids)}      # tiktoken id -> our id
ids = np.array([to_small[t] for t in tokens])                      # the whole file in our ids
vocab = len(used_ids)                                              # 4841
# to go back later: used_ids[our_id] is the tiktoken id, then encoding.decode([...])

d = 128      # numbers per token
max_T = 64   # tokens the model sees at once

rng = np.random.default_rng(0)                       # seeded so you get the same random numbers every run
tok_table = rng.normal(0, 0.02, (vocab, d))          # (4841, 128): one row per token, its meaning
pos_table = rng.normal(0, 0.02, (max_T, d))          # (64, 128): one row per position, where it sits

chunk = ids[:max_T]                                  # the first 64 tokens of the file
x = tok_table[chunk] + pos_table[:len(chunk)]        # (64, 128): each token's row plus its position's row
print(vocab, x.shape)
