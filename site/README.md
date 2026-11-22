# GPT by Hand (the website)

A static Next.js site that hosts the model trained on RunPod and explains the NumPy and CuPy code.

- `/`       the playground: the trained model runs **in the visitor's browser** (no server), plus the training curve
- `/numpy`  `main_functionized.py`, section by section, with copy buttons
- `/cuda`   every change from NumPy to CuPy, plus `train_tinystories.py` (the script that trained the model)

## Commands (from this folder)

    npm run dev        # local dev server at http://localhost:3000
    npm run build      # builds the static site into out/
    npm start          # serves out/ locally

Both `dev` and `build` first copy the latest `../main_functionized.py`, `../main_functionized_cuda.py` and
`../runpod/train_tinystories.py` into `public/code/`.

## How the model gets onto the site

    ../.venv/bin/python scripts/export_model.py   # best_model.pkl -> public/model/weights.bin (float16) + manifest.json
    node scripts/test_worker.mjs                  # checks the javascript model gives the same answers as numpy

`public/gpt-worker.js` is the model in plain JavaScript (same math as `forward()` in train_tinystories.py, with a
key/value cache so each new token costs one row instead of the whole window). The tokenizer is `js-tiktoken`
(o200k_base, the same as Python's tiktoken).

## Text on the pages

The walkthrough text is in `content/numpy-walkthrough.json` (sections point at line ranges of main_functionized.py)
and `content/cuda-walkthrough.json` (before/after snippets copied word for word from the two files). If you edit
main_functionized.py, check that the line ranges still line up.

## Deploying

It's a plain static site (`out/`), so it can go on Vercel, Netlify, GitHub Pages or any web host.
On Vercel: `npx vercel` from this folder (framework: Next.js).
