# For Claude: read this first

## The goal

Asher is building **a whole GPT in one Python file, using only plain Python and NumPy**, and writing it
**by hand** to learn how every piece works. The finished file reads `stories.txt` (1,254 short children's
stories, about 1 MB), turns it into tokens, trains a small transformer on it with forward and backward
passes written out manually (no PyTorch, no autograd), and then writes new stories from a prompt like
`"Once upon a time, there was a"`.

The point is **understanding**, not just having a working model. A working `gpt.py` already exists (see
references below). Asher is rebuilding it step by step in `main.py` so they understand every line.

## Who edits what

| File | Who edits it | What it's for |
|---|---|---|
| `../main.py` | **Asher only. Claude never edits it.** | Asher's own build. Read it to see where they're at. |
| `../main2.py` | **Claude** | The step-by-step build, fully commented: one window (64, 128), no batch, each step run and printed so its output can be checked. Show the next step here. Asher types what they want into `main.py` themselves. |
| `../main3.py` | **Claude** | **The complete program that trains**, written in the style of Asher's `main.py`: plain top-level code, short `#we ...` comments, nothing fancy. Six blocks. **No separate attention/mlp/block functions**: each block's code is written inline in one loop (block1 → block2 → … → block6), and the backward pass is the same loop in reverse. The only helpers are the small math ones (`layer_norm`, `linear`, `softmax`, `gelu` and their backwards). Running it trains (about 3,000 steps) and then writes 3 stories. No save/load. If you change the math, gradient-check it on a tiny model first. |
| `../main4.py` | **Claude** | An exact copy of Asher's current `main.py` plus **just the next step**, in Asher's style (short `#` comments, inline, no new functions). When Asher asks for the next step, re-copy `main.py` and add one step. |
| `../main_functionized.py` | **Claude** | Asher's `main.py`, **with Asher's own comments kept word for word**, turned into a `forward(window)` function so it's ready for backprop. All learned numbers are in one `params` dict (block1. … block6.), and every step returns a cache. It works on **one window at a time** (64, 128), like `main.py`, so Asher's shape comments stay true. No batch dimension yet. Same output as `main3.py`'s gradient-checked forward. |
| `../main6.py` | **Claude** | `main_functionized.py` + **the backward pass, built one step at a time** (like `main4.py`). Each time Asher says keep going, add the next backward step in Asher's comment style, with a small numerical check where it helps. Step 1 is done: logits gradient = softmax − 1 at the right answer, ÷ 64. Next: back through the head linear. The finished version to compare against is `main3.py`'s backward. |
| `README.md` (this file) | Claude | Keep the "Where Asher is" section up to date when a step is done or a decision changes. |
| `../lessons/` | Claude, when asked | One explainer page per step (e.g. `bpe.html`). |

When Asher asks for the next step, put it in `main2.py` and explain it in chat. `main3.py` is the finished
reference to compare against. Don't paste it into `main.py`.
If Asher asks "is my thing okay?", read `main.py`, run it, and answer. Only edit `main.py` if they
explicitly say to edit **main.py**.

## Reference material

- **`../gpt.txt`**: the finished one-file GPT (identical to `~/Desktop/numpy-gpt-explained/gpt.py`). It's
  the answer key. Built top to bottom in numbered steps, with every line commented:
  - Steps **1–21** are the forward pass: text, tokens, embeddings, LayerNorm, Q/K/V, attention scores,
    causal mask, softmax, weighted values, heads combined, residual, LayerNorm, MLP up, GELU, MLP down,
    residual, a stack of blocks, logits, probabilities, sampling.
  - Steps **L1–L8** are learning: batches, cross-entropy loss, backward through the head, blocks and
    embeddings, Adam, the training loop, and `main()`.
  - Sizes: `D=128`, `N_LAYERS=4`, `N_HEADS=4`, `T=64` (context), `B=16`, `STEPS=3000`, `LR=3e-3`.
    That's about 0.93M learned numbers. Its training run goes from loss 6.26 to a test loss of about 2.2.
  - Style: no classes, just functions and dicts of arrays. `model = {"p": {name: array}, ...}`.
- **The website**: https://numpy-gpt-explained.vercel.app ("Inside the NumPy GPT"). It walks through the
  same steps in the same order as `gpt.txt`. The source is at `~/Desktop/numpy-gpt-explained/` (`index.html`,
  built by `build.py` and `visuals.py`).
- **`~/Desktop/numpy-gpt-build-plan.txt`**: a no-code build plan in stages (NumPy warm-up, math functions,
  layers with gradient checks, the whole model, tokenizer, training, generation), with book section
  numbers in brackets.
- **`~/Desktop/build-a-gpt-from-scratch.pdf`**: the book those section numbers refer to.
- **`../lessons/bpe.html`**: the explainer page for step 2 (byte pair encoding).

## Where Asher is (keep this updated)

- **Step 1 (text): done.** Reads `stories.txt`.
- **Step 2 (tokens): done, but different from `gpt.txt`.** Asher first wrote a character-level BPE by hand
  (they call it "cpe", backed up in Claude's scratchpad), then switched to **OpenAI's `tiktoken` with
  `o200k_base`** (the GPT-4o encoding).
  - `stories.txt` becomes **239,412 tokens**, and `decode(encode(text)) == text` exactly.
  - `o200k_base` has 200,019 ids, but the stories only use **4,841 different ones**. The plan is to
    renumber them 0..4840 so the token table has 4,841 rows instead of 200,019.
- **Step 3 (position + token embedding): in Asher's `main.py`** (with their own names: `vocab_size`,
  `max_tokens`, `tok_table`, `pos_table`). `main2.py` uses clearer names: `embedding_size`, `context_length`,
  `token_embedding_table`, `position_embedding_table`, `window`, `vectors`.
- **Step 4 (LayerNorm): written in `main2.py`, not yet in `main.py`.** It's forward only there.
- **Step 5 (Q, K, V): written in `main2.py`, not yet in `main.py`.** One `linear` (128 -> 384),
  `np.split` into queries/keys/values, then `split_into_heads` gives 4 heads of 32: shape (4, 64, 32).
  It works on one window (64, 128), with no batch dimension yet, unlike `gpt.txt`'s (B, T, d).
- **Next in the step-by-step build (`main2.py`): step 6 (query · key scores).**
- **`main3.py` has everything (steps 1–21, L1–L8) and trains.** First full run (4 blocks, about 2.05M numbers,
  6.4 minutes): train loss 8.52 → about 2.0, but **test loss 3.54**, so it's overfitting. The 4,841-token head and
  table are big for 215K training tokens. It writes coherent little stories anyway. It's now 6 blocks, at Asher's request.
  6-block run (2.44M numbers): train loss about 2.0, **test loss 3.51**. Same gap, same quality of stories.

## How Asher likes to work

- Explain the **process**, with small concrete examples (`"the" -> ['t', 'h', 'e'] -> ['t', 'he']`).
  Asher is learning, so the "why" matters as much as the code.
- **Lots of comments** in code, in plain words, saying what each line does and why.
- Plain functions and plain data, matching `gpt.txt`'s style. Asher sometimes asks for no library helpers,
  so the mechanics are visible.
- Asher's messages are fast and casual, and sometimes dictated (e.g. "cpe" means BPE, "main to" means
  main2). Figure out the intent instead of asking.
- Point out where Asher's version differs from `gpt.txt` and what that changes (e.g. tiktoken means a
  much bigger vocabulary than `gpt.txt`'s 512).

## Running things

- Use the project's virtual environment: `.venv/bin/python main.py` (or `source .venv/bin/activate` once
  per terminal). It has `tiktoken` and `numpy`. Homebrew's `python3` blocks global `pip install`.
- `main.py` uses `Path.cwd()`, so it must be run from the `gpt-one-file` folder. `main2.py` and `main3.py`
  use `Path(__file__).parent` and work from anywhere. (They live in the project folder, next to `main.py`.)
- This isn't a git repo, so there's no history. Back up before big changes.
