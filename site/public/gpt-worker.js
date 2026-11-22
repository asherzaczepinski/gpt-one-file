// the trained model, in plain javascript, running in a background thread (a web worker) so the page never freezes.
// same math as forward() in train_tinystories.py: embeddings -> 6 blocks (layer norm, attention, mlp) -> final layer norm -> head.
// the one difference: python re-reads the whole window for every new token. here each token is read once, and its keys
// and values are kept in a cache, so a new token only needs its own row (about 100x less work). same answers.

const GPT = {};
self.GPT = GPT;

// float16 -> float32, with a lookup table for all 65,536 possible float16 bit patterns
function halfTable() {
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
    if (exponent === 0) table[h] = sign * Math.pow(2, -14) * (fraction / 1024);
    else if (exponent === 31) table[h] = fraction ? NaN : sign * Infinity;
    else table[h] = sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
  }
  return table;
}

GPT.loadFromBuffers = function (manifest, buffer) {
  const halves = new Uint16Array(buffer), table = halfTable(), W = {};
  for (const t of manifest.tensors) {
    const size = t.shape.reduce((a, b) => a * b, 1), a = new Float32Array(size);
    for (let i = 0; i < size; i++) a[i] = table[halves[t.offset + i]];
    W[t.name] = a;
  }
  const s = manifest.settings;
  GPT.m = {
    W, D: s.embedding_size, T: s.context_length, H: s.number_of_heads, L: s.number_of_blocks,
    hs: s.embedding_size / s.number_of_heads, V: manifest.vocab_size,
  };
};

// out = x @ W + b, where W is (inDim, outDim) stored row by row
function linear(x, W, b, inDim, outDim, out) {
  out.set(b);
  for (let i = 0; i < inDim; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = i * outDim;
    for (let j = 0; j < outDim; j++) out[j] += xi * W[row + j];
  }
  return out;
}

// mean 0, spread 1, then scale and shift (numpy's var = average squared distance, tiny = 1e-5)
function layerNorm(x, scale, shift, out) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) * (x[i] - mean);
  const spread = Math.sqrt(variance / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = ((x[i] - mean) / spread) * scale[i] + shift[i];
  return out;
}

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

// keys and values of every token read so far, for every block
GPT.newCache = function () {
  const { D, T, L } = GPT.m, blocks = [];
  for (let i = 0; i < L; i++) blocks.push({ keys: new Float32Array(T * D), values: new Float32Array(T * D) });
  return { blocks, length: 0 };
};

// read one token at position pos. returns the scores for the next token (or null when wantLogits is false)
GPT.step = function (token, cache, wantLogits) {
  const { W, D, T, H, hs, L, V } = GPT.m, pos = cache.length;
  if (pos >= T) throw new Error("cache full");
  const x = new Float32Array(D), normed = new Float32Array(D), qkv = new Float32Array(3 * D);
  const gathered = new Float32Array(D), attn = new Float32Array(D), wide = new Float32Array(4 * D), mlp = new Float32Array(D);
  const tok = W.token_embedding_table, posTable = W.position_embedding_table;
  for (let d = 0; d < D; d++) x[d] = tok[token * D + d] + posTable[pos * D + d];
  const scores = new Float32Array(T), root = Math.sqrt(hs);

  for (let i = 0; i < L; i++) {
    const b = "block" + (i + 1) + ".", c = cache.blocks[i];
    layerNorm(x, W[b + "layer_norm_scale"], W[b + "layer_norm_shift"], normed);
    linear(normed, W[b + "qkv_weights"], W[b + "qkv_bias"], D, 3 * D, qkv);
    // queries are qkv[0:D], keys qkv[D:2D], values qkv[2D:3D]. save this token's key and value
    c.keys.set(qkv.subarray(D, 2 * D), pos * D);
    c.values.set(qkv.subarray(2 * D, 3 * D), pos * D);
    // every head: score against every token so far (only the past exists here, so no mask needed), softmax, mix the values
    for (let h = 0; h < H; h++) {
      const o = h * hs;
      let best = -Infinity;
      for (let t = 0; t <= pos; t++) {
        let s = 0;
        for (let d = 0; d < hs; d++) s += qkv[o + d] * c.keys[t * D + o + d];
        scores[t] = s / root;
        if (scores[t] > best) best = scores[t];
      }
      let total = 0;
      for (let t = 0; t <= pos; t++) { scores[t] = Math.exp(scores[t] - best); total += scores[t]; }
      for (let d = 0; d < hs; d++) gathered[o + d] = 0;
      for (let t = 0; t <= pos; t++) {
        const w = scores[t] / total;
        for (let d = 0; d < hs; d++) gathered[o + d] += w * c.values[t * D + o + d];
      }
    }
    linear(gathered, W[b + "output_weights"], W[b + "output_bias"], D, D, attn);
    for (let d = 0; d < D; d++) x[d] += attn[d];
    layerNorm(x, W[b + "layer_norm2_scale"], W[b + "layer_norm2_shift"], normed);
    linear(normed, W[b + "expand_weights"], W[b + "expand_bias"], D, 4 * D, wide);
    for (let j = 0; j < 4 * D; j++) wide[j] = gelu(wide[j]);
    linear(wide, W[b + "shrink_weights"], W[b + "shrink_bias"], 4 * D, D, mlp);
    for (let d = 0; d < D; d++) x[d] += mlp[d];
  }
  cache.length = pos + 1;
  if (!wantLogits) return null;
  layerNorm(x, W.final_layer_norm_scale, W.final_layer_norm_shift, normed);
  return linear(normed, W.head_weights, W.head_bias, D, V, new Float32Array(V));
};

// read a list of tokens from position 0, return the scores after the last one
GPT.prefill = function (ids, cache) {
  let logits = null;
  ids.forEach((id, i) => { logits = GPT.step(id, cache, i === ids.length - 1); });
  return logits;
};

// a small seeded random number generator, so the same seed gives the same story
function rng(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// scores -> temperature -> keep the top k -> softmax. same as next_token_probabilities in write_story.py
GPT.probabilities = function (logits, temperature, topK) {
  const n = logits.length, scaled = new Float32Array(n);
  for (let i = 0; i < n; i++) scaled[i] = logits[i] / temperature;
  const cutoff = scaled.slice().sort()[n - Math.min(topK, n)];
  let best = -Infinity;
  for (let i = 0; i < n; i++) if (scaled[i] >= cutoff && scaled[i] > best) best = scaled[i];
  const p = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) if (scaled[i] >= cutoff) { p[i] = Math.exp(scaled[i] - best); total += p[i]; }
  for (let i = 0; i < n; i++) p[i] /= total;
  return p;
};

// ---- messages from the page ----
let stopRequested = false;

async function loadModel(base) {
  const manifest = await (await fetch(base + "manifest.json")).json();
  const response = await fetch(base + "weights.bin");
  const total = Number(response.headers.get("content-length")) || manifest.total * 2;
  const reader = response.body.getReader(), parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    self.postMessage({ type: "progress", loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const part of parts) { bytes.set(part, at); at += part.length; }
  GPT.loadFromBuffers(manifest, bytes.buffer);
  self.postMessage({ type: "ready" });
}

async function generate({ ids, tokens, temperature, topK, seed }) {
  stopRequested = false;
  const { T } = GPT.m, random = rng(seed), started = performance.now();
  let history = ids.slice(-T), cache = GPT.newCache();
  let logits = GPT.prefill(history, cache);
  for (let n = 0; n < tokens; n++) {
    const p = GPT.probabilities(logits, temperature, topK);
    // roll the weighted die: walk through the chances until the random number runs out
    let r = random(), choice = -1, last = 0;
    for (let i = 0; i < p.length; i++) {
      if (p[i] === 0) continue;
      last = i;
      r -= p[i];
      if (r <= 0) { choice = i; break; }
    }
    if (choice < 0) choice = last;   // rounding left a tiny bit over
    const top = [];
    for (let i = 0; i < p.length; i++) if (p[i] > 0) top.push(i);
    top.sort((a, b) => p[b] - p[a]);
    self.postMessage({ type: "token", id: choice, p: p[choice], top: top.slice(0, 8).map((i) => [i, p[i]]) });
    history.push(choice);
    if (cache.length < T) {
      logits = GPT.step(choice, cache, true);
    } else {
      // the window is full (128 tokens): start over from the last 96, so there's room for 32 more before the next re-read
      history = history.slice(-96);
      cache = GPT.newCache();
      logits = GPT.prefill(history, cache);
    }
    await new Promise((r) => setTimeout(r, 0));   // lets a "stop" message get through
    if (stopRequested) break;
  }
  self.postMessage({ type: "done", seconds: (performance.now() - started) / 1000 });
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "load") loadModel(msg.base).catch((err) => self.postMessage({ type: "error", message: String(err) }));
  if (msg.type === "generate") generate(msg).catch((err) => self.postMessage({ type: "error", message: String(err) }));
  if (msg.type === "stop") stopRequested = true;
};
