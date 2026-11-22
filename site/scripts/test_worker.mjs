// checks the javascript model against the numpy model: same prompt -> same next-token scores
//   node scripts/test_worker.mjs   (from the site folder, after export_model.py)
import fs from "node:fs";
import vm from "node:vm";

const self = { postMessage() {} };
vm.runInNewContext(fs.readFileSync("public/gpt-worker.js", "utf8"), { self, fetch, performance, setTimeout, console, Math });
const GPT = self.GPT;

const manifest = JSON.parse(fs.readFileSync("public/model/manifest.json", "utf8"));
const file = fs.readFileSync("public/model/weights.bin");
GPT.loadFromBuffers(manifest, file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));

const fixture = JSON.parse(fs.readFileSync("scripts/fixture.json", "utf8"));
let t0 = performance.now();
const logits = GPT.prefill(fixture.ids, GPT.newCache());
const prefillMs = performance.now() - t0;

let worst = 0;
fixture.top_ids.forEach((id, i) => { worst = Math.max(worst, Math.abs(logits[id] - fixture.top_logits[i])); });
let jsTop = 0;
for (let i = 0; i < logits.length; i++) if (logits[i] > logits[jsTop]) jsTop = i;
console.log(`top-20 logits: biggest difference from numpy ${worst.toFixed(4)} (float16 alone gave up to ~0.035)`);
console.log(`top token: js ${jsTop}, numpy ${fixture.top_ids[0]} -> ${jsTop === fixture.top_ids[0] ? "same" : "DIFFERENT"}`);

// speed: 100 tokens one at a time with the cache
const cache = GPT.newCache();
t0 = performance.now();
for (let i = 0; i < 100; i++) GPT.step(fixture.ids[i % fixture.ids.length], cache, true);
console.log(`prefill ${fixture.ids.length} tokens: ${prefillMs.toFixed(0)} ms, then ${(100 / ((performance.now() - t0) / 1000)).toFixed(1)} tokens/sec`);
if (worst > 0.1 || jsTop !== fixture.top_ids[0]) process.exit(1);
