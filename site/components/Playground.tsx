"use client";
import { Fragment, useEffect, useRef, useState } from "react";
import type { Tiktoken } from "js-tiktoken/lite";

type Step = { id: number; piece: string; p: number; top: [string, number][] };
type Tok = { piece: string; known: boolean };

const PRESETS = [
  "Once upon a time, there was a",
  "Tom and Sam went to the park.",
  "The little cat was sad because",
  "Lily found a shiny key in the garden.",
  "One day, a big dragon",
  'Mom said, "',
];
// the marker between stories. the training text had it as plain characters, so the model writes it as 7 ordinary tokens
// (<, |, end, of, text, |, >) and the page joins them back into one "end of story" divider
const END = "<|endoftext|>";

const visible = (s: string) => s.replace(/ /g, "␣").replace(/\n/g, "↵");
const pct = (p: number) => (p >= 0.995 ? "100" : (p * 100).toFixed(p < 0.1 ? 1 : 0)) + "%";
const shade = (p: number) => `color-mix(in srgb, var(--accent) ${Math.round((1 - p) * 42)}%, transparent)`;

export default function Playground() {
  const [prompt, setPrompt] = useState(PRESETS[0]);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [length, setLength] = useState(150);
  const [seed, setSeed] = useState(0);
  const [shading, setShading] = useState(true);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "writing">("idle");
  const [progress, setProgress] = useState({ loaded: 0, total: 0 });
  const [steps, setSteps] = useState<Step[]>([]);
  const [promptText, setPromptText] = useState("");
  const [promptTokens, setPromptTokens] = useState<Tok[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [speed, setSpeed] = useState("");
  const [error, setError] = useState("");

  const worker = useRef<Worker | null>(null);
  const enc = useRef<Tiktoken | null>(null);
  const usedIds = useRef<number[]>([]);
  const toSmall = useRef<Map<number, number>>(new Map());
  const pending = useRef(false);
  const started = useRef(0);
  const count = useRef(0);
  const writeLatest = useRef<() => void>(() => {});   // the newest write(), so a write queued during loading uses the newest settings

  const decode = (smallId: number) => enc.current!.decode([usedIds.current[smallId]]);

  function tokenize(text: string): { tokens: Tok[]; ids: number[] } {
    const tokens: Tok[] = [], ids: number[] = [];
    for (const big of enc.current!.encode(text, [], [])) {   // plain text, like encode_ordinary in python
      const small = toSmall.current.get(big);
      tokens.push({ piece: enc.current!.decode([big]), known: small !== undefined });
      if (small !== undefined) ids.push(small);
    }
    return { tokens, ids };
  }

  // the prompt as tokens, once the tokenizer is loaded
  useEffect(() => {
    if (enc.current) setPromptTokens(tokenize(prompt).tokens);
  }, [prompt, status]);

  useEffect(() => () => worker.current?.terminate(), []);

  async function load() {
    setStatus("loading");
    setError("");
    try {
      const [lite, ranks, manifest] = await Promise.all([
        import("js-tiktoken/lite"),
        import("js-tiktoken/ranks/o200k_base"),
        fetch("/model/manifest.json").then((r) => r.json()),
      ]);
      enc.current = new lite.Tiktoken(ranks.default);
      usedIds.current = manifest.used_ids;
      toSmall.current = new Map(manifest.used_ids.map((big: number, small: number) => [big, small]));
    } catch (e) {
      setError("Couldn't load the tokenizer: " + String(e));
      setStatus("idle");
      return;
    }
    const w = new Worker("/gpt-worker.js");
    worker.current = w;
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") setProgress({ loaded: m.loaded, total: m.total });
      else if (m.type === "ready") {
        setStatus("ready");
        if (pending.current) {
          pending.current = false;
          writeLatest.current();
        }
      } else if (m.type === "token") {
        count.current += 1;
        const step: Step = { id: m.id, piece: decode(m.id), p: m.p, top: m.top.map(([id, p]: [number, number]) => [decode(id), p]) };
        setSteps((s) => [...s, step]);
        const secs = (performance.now() - started.current) / 1000;
        setSpeed(`${count.current} tokens · ${(count.current / secs).toFixed(0)} per second`);
      } else if (m.type === "done") {
        setStatus("ready");
        setSpeed(`${count.current} tokens in ${m.seconds.toFixed(1)} s (${(count.current / m.seconds).toFixed(0)} per second)`);
      } else if (m.type === "error") {
        setError(m.message);
        setStatus(enc.current ? "ready" : "idle");
      }
    };
    w.postMessage({ type: "load", base: "/model/" });
  }

  function write() {
    if (!worker.current || !enc.current) {
      pending.current = true;
      if (status === "idle") load();
      return;
    }
    const { tokens, ids } = tokenize(prompt);
    setPromptTokens(tokens);
    setPromptText(tokens.filter((t) => t.known).map((t) => t.piece).join(""));
    setSteps([]);
    setSelected(null);
    setSpeed("");
    count.current = 0;
    started.current = performance.now();
    setStatus("writing");
    const startIds = ids.length ? ids : tokenize(END).ids;   // an empty prompt starts as if a story just ended
    worker.current.postMessage({ type: "generate", ids: startIds, tokens: length, temperature, topK, seed });
  }

  useEffect(() => { writeLatest.current = write; });

  // the end-of-story marker comes out in pieces that don't line up with it (like "<", "|", "end", ... , "|>\n"),
  // so mark which characters of the whole story are marker, then show each token minus those, with a divider where a marker ends
  const display: { before: string; divider: boolean; after: string }[] = [];
  {
    let text = "";
    const starts = steps.map((st) => { const at = text.length; text += st.piece; return at; });
    const isMarker = new Uint8Array(text.length), endsAt = new Set<number>();
    for (let at = text.indexOf(END); at >= 0; at = text.indexOf(END, at + 1)) {
      isMarker.fill(1, at, at + END.length);
      endsAt.add(at + END.length - 1);
    }
    steps.forEach((st, i) => {
      const d = { before: "", divider: false, after: "" };
      for (let c = starts[i]; c < starts[i] + st.piece.length; c++) {
        if (!isMarker[c]) { if (d.divider) d.after += text[c]; else d.before += text[c]; }
        if (endsAt.has(c)) d.divider = true;
      }
      display.push(d);
    });
  }

  const busy = status === "loading" || status === "writing";
  const sel = selected !== null ? steps[selected] : null;

  return (
    <div className="card playground">
      <label className="lab" htmlFor="prompt">Start of the story</label>
      <textarea
        id="prompt"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !busy) write(); }}
      />
      <div className="chips">
        {PRESETS.map((p) => (
          <button key={p} className="chip" onClick={() => setPrompt(p)}>{p}</button>
        ))}
      </div>

      {promptTokens.length > 0 && (
        <>
          <div className="lab" style={{ marginTop: 14 }}>
            How the model sees it <span className="lab-note">(one box = one token. crossed out = never seen in training, skipped)</span>
          </div>
          <div className="tokens">
            {promptTokens.map((t, i) => (
              <span key={i} className={"tok" + (t.known ? "" : " unknown")}>{visible(t.piece)}</span>
            ))}
          </div>
        </>
      )}

      <div className="knobs">
        <div>
          <label className="lab" htmlFor="temp">Temperature <span className="val">{temperature.toFixed(2)}</span></label>
          <input id="temp" type="range" min={0.1} max={2} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} />
          <div className="hint">low = safe, high = wild</div>
        </div>
        <div>
          <label className="lab" htmlFor="topk">Top-k <span className="val">{topK}</span></label>
          <input id="topk" type="range" min={1} max={200} step={1} value={topK} onChange={(e) => setTopK(+e.target.value)} />
          <div className="hint">pick only from the k likeliest</div>
        </div>
        <div>
          <label className="lab" htmlFor="len">Length <span className="val">{length}</span></label>
          <input id="len" type="range" min={10} max={400} step={10} value={length} onChange={(e) => setLength(+e.target.value)} />
          <div className="hint">new tokens to write</div>
        </div>
        <div>
          <label className="lab" htmlFor="seed">Seed</label>
          <div className="seedrow">
            <input id="seed" type="number" value={seed} onChange={(e) => setSeed(Math.floor(+e.target.value) || 0)} />
            <button className="copy" onClick={() => setSeed(Math.floor(Math.random() * 100000))} aria-label="random seed">🎲</button>
          </div>
          <div className="hint">same seed = same story</div>
        </div>
      </div>

      <div className="row" style={{ marginTop: 16 }}>
        <button className="btn solid" onClick={write} disabled={busy}>
          {status === "idle" ? "Load the model and write" : status === "loading" ? "Loading…" : "Write"}
        </button>
        <button className="btn ghost" onClick={() => worker.current?.postMessage({ type: "stop" })} disabled={status !== "writing"}>
          Stop
        </button>
        <label className="toggle">
          <input type="checkbox" checked={shading} onChange={(e) => setShading(e.target.checked)} /> shade by how sure it was
        </label>
      </div>

      {status === "loading" && (
        <div className="progress">
          <div className="track"><div className="fill" style={{ width: `${progress.total ? (100 * progress.loaded) / progress.total : 2}%` }} /></div>
          <span>
            Downloading the model: {(progress.loaded / 1e6).toFixed(1)} of {progress.total ? (progress.total / 1e6).toFixed(1) : "34.8"} MB.
            It runs on your computer, so this only happens once per visit.
          </span>
        </div>
      )}
      {status === "idle" && !error && (
        <p className="hint" style={{ marginTop: 10 }}>The first click downloads the model (35 MB) into your browser. After that, every story is written on your own device.</p>
      )}
      {error && <p className="err">{error}</p>}

      <div className="story" aria-live="polite">
        {steps.length === 0 && status !== "writing" ? (
          <span className="placeholder">The story will appear here.</span>
        ) : (
          <>
            <span className="prompt-text">{promptText}</span>
            {steps.map((s, i) => {
              const d = display[i];
              const word = (t: string) => (
                <span
                  className={"g" + (selected === i ? " sel" : "")}
                  style={{ background: shading ? shade(s.p) : "transparent" }}
                  onClick={() => setSelected(i)}
                >
                  {t}
                </span>
              );
              return (
                <Fragment key={i}>
                  {d.before && word(d.before)}
                  {d.divider && <span className="end" onClick={() => setSelected(i)}>· end of story ·</span>}
                  {d.after && word(d.after.replace(/^\n/, ""))}
                </Fragment>
              );
            })}
          </>
        )}
      </div>
      <div className="speed">{speed}</div>

      <div className="inspect">
        <div className="lab">What it was choosing between</div>
        {sel ? (
          <>
            <p className="hint">
              Token {selected! + 1}: it picked “{sel.piece === END ? "end of story" : visible(sel.piece)}” with a {pct(sel.p)} chance.
              {sel.p < 0.2 ? " A long shot: the dice got lucky." : sel.p > 0.8 ? " It was pretty sure about this one." : ""}
            </p>
            {sel.top.map(([w, p], i) => (
              <div key={i} className={"bar-row" + (w === sel.piece ? " chosen" : "")}>
                <span className="w">{w === END ? "[end of story]" : visible(w)}</span>
                <span className="track"><span className="fill" style={{ width: `${(p * 100).toFixed(1)}%` }} /></span>
                <span className="pc">{pct(p)}</span>
              </div>
            ))}
          </>
        ) : (
          <p className="hint">Click any word in the story to see the other tokens it could have written there, and their chances.</p>
        )}
      </div>
    </div>
  );
}
