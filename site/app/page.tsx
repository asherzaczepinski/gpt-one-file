import Link from "next/link";
import Playground from "@/components/Playground";
import LossChart from "@/components/LossChart";
import { readCode } from "@/lib/code";

const codeLines = (name: string) => readCode(name).split("\n").filter((l) => l.trim() && !l.trim().startsWith("#")).length;

export default function Home() {
  return (
    <main>
      <section className="wrap hero">
        <div className="eyebrow">Built from scratch by a high schooler</div>
        <h1>A GPT written by hand in NumPy, trained for 93 cents</h1>
        <p className="lede">
          No PyTorch, no autograd. Every line of it, from the tokens to attention to backprop, was written by hand to learn how
          ChatGPT actually works. Then the same code ran on a rented GPU. The model below is the result, and it runs right here in
          your browser.
        </p>
        <div className="stats">
          <div className="stat"><div className="v">17.4M</div><div className="l">learned numbers</div></div>
          <div className="stat"><div className="v">97.5M</div><div className="l">tokens of stories read</div></div>
          <div className="stat"><div className="v">$0.93</div><div className="l">to train, on one rented GPU</div></div>
          <div className="stat"><div className="v">1.545</div><div className="l">test loss (random guessing: 10.1)</div></div>
        </div>
      </section>

      <section className="wrap" id="try">
        <h2 style={{ marginTop: "1.2em" }}>Try it</h2>
        <p>
          Give it the start of a story. It writes one token at a time: it scores all 24,596 tokens it knows, turns the scores into
          chances, and rolls a weighted die. Click any word it writes to see what else it almost picked.
        </p>
        <Playground />
        <p className="hint" style={{ marginTop: 12 }}>
          It only ever read simple children&apos;s stories (the TinyStories dataset), so that&apos;s all it can write. It has no facts
          and can&apos;t chat: ask it a question and you&apos;ll get a story. It also only sees the last 128 tokens (about 100 words),
          so long stories can drift.
        </p>
      </section>

      <section className="wrap">
        <h2>How it learned</h2>
        <p>
          The <b>loss</b> measures how surprised the model is by the real next token. Lower is better. It trained for 222 minutes on an
          NVIDIA RTX 2000 Ada rented on RunPod for $0.24 an hour, reading 27,000 tokens a second.
        </p>
        <LossChart />
      </section>

      <section className="wrap">
        <h2>Same prompt, two models</h2>
        <p>
          The first version trained on a laptop, on about 1 MB of stories. It started memorizing them after about 6 minutes. The fix
          wasn&apos;t a faster computer, it was more text: the GPU version read about 450 times more.
        </p>
        <div className="compare">
          <div className="card">
            <div className="tag">Laptop · 2.4M numbers · test loss 3.48</div>
            <p>
              Tom and Sam went to the park. They ran to slide and play again. They all played together and had fun on the slide. They
              felt good at Tom was happy and happy. Tim was not happy that he helped their play with the blocks.
            </p>
          </div>
          <div className="card">
            <div className="tag">GPU · 17.4M numbers · test loss 1.545</div>
            <p>
              Tom and Sam went to the park. They played on the swings and slide. They had fun. But soon, they got very tired. Tom said,
              &quot;Sam, let&apos;s go home now. We need to rest.&quot; Sam agreed. They sat down under a big tree. They took a long nap. At
              the end of the day, they were both better.
            </p>
          </div>
        </div>
        <p className="hint" style={{ marginTop: 10 }}>
          The two losses aren&apos;t a perfect comparison (each model has its own vocabulary), but the stories show the difference.
        </p>
      </section>

      <section className="wrap">
        <h2>How it works, in six steps</h2>
        <ol className="steps-run">
          <li><b>Tokens.</b> The text is cut into pieces with the same tokenizer as ChatGPT (o200k). &quot;Once upon a time&quot; is 4 tokens.</li>
          <li><b>Embeddings.</b> Each token gets a list of 256 learned numbers, plus another list for its position in the window.</li>
          <li><b>Attention.</b> Every token asks the tokens before it which ones matter (queries, keys, values), in 8 heads at once, and mixes in their information.</li>
          <li><b>Thinking (the MLP).</b> Each token&apos;s numbers are widened to 1,024, bent with GELU, and squeezed back to 256.</li>
          <li><b>Six blocks.</b> Attention + MLP is one block. The model stacks six, then scores every possible next token.</li>
          <li><b>Learning.</b> The loss says how wrong the guess was, backprop (written by hand) works out how to nudge each of the 17.4 million numbers, and Adam nudges them.</li>
        </ol>
      </section>

      <section className="wrap">
        <h2>Read the code</h2>
        <p>Every line is on this site, explained, ready to copy.</p>
        <div className="links">
          <Link href="/numpy" className="card">
            <h3>1. The NumPy version</h3>
            <p>The whole GPT in one file, {codeLines("main_functionized.py")} lines of code plus lots of comments. Trains on a laptop in about 6 minutes.</p>
            <span className="go">Walk through it →</span>
          </Link>
          <Link href="/cuda" className="card">
            <h3>2. The GPU version (CuPy)</h3>
            <p>The same math, made to run on an NVIDIA GPU: every change explained, plus the exact script that trained the model above.</p>
            <span className="go">See what changed →</span>
          </Link>
        </div>
      </section>
    </main>
  );
}
