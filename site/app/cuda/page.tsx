import type { Metadata } from "next";
import CodeBlock from "@/components/CodeBlock";
import CopyButton from "@/components/CopyButton";
import { Paragraphs, rich } from "@/lib/rich";
import { dedent, readCode } from "@/lib/code";
import walkthrough from "@/content/cuda-walkthrough.json";

export const metadata: Metadata = {
  title: "The GPU version",
  description: "The same hand-written GPT, ported to NVIDIA GPUs with CuPy: every change explained, plus the script that trained the model.",
};

const CUDA = "main_functionized_cuda.py";
const TRAINER = "train_tinystories.py";
const slug = (t: string) => t.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");

const RUN = `# on a machine with an NVIDIA GPU (CUDA 12)
pip install cupy-cuda12x tiktoken numpy

# 400 MB of TinyStories (about 100 million tokens)
curl -L https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt | head -c 400000000 > tinystories.txt

# train for 60 minutes, then save out/best_model.pkl and write 4 stories
TEXT_FILE=tinystories.txt TRAIN_MINUTES=60 python ${TRAINER}`;

export default function CudaPage() {
  const cuda = readCode(CUDA);
  const trainer = readCode(TRAINER);
  const w = walkthrough;

  return (
    <main className="wide">
      <section className="hero" style={{ maxWidth: 820 }}>
        <div className="eyebrow">The code · part 2</div>
        <h1>The GPU version</h1>
        <Paragraphs items={w.intro} />
        <div className="row" style={{ marginTop: 10 }}>
          <a className="btn solid" href={`/code/${CUDA}`} download>Download {CUDA}</a>
          <a className="btn ghost" href={`/code/${TRAINER}`} download>Download {TRAINER}</a>
          <CopyButton text={cuda} label={`Copy ${CUDA}`} />
        </div>
      </section>

      <div className="page-grid" style={{ marginTop: 30 }}>
        <aside className="toc">
          <div className="t">On this page</div>
          <a href="#ideas">The big ideas</a>
          <a href="#changes">Every change, before and after</a>
          {w.changes.map((c, i) => (
            <a key={i} href={`#${slug(c.title)}`} style={{ paddingLeft: 20 }}>{c.title}</a>
          ))}
          <a href="#trainer">The script that trained the model</a>
          <a href="#run">Run it yourself</a>
          <a href="#whole-files">The whole files</a>
        </aside>

        <div style={{ maxWidth: 900, minWidth: 0 }}>
          <h2 id="ideas" style={{ marginTop: 0 }}>The big ideas</h2>
          <div className="concepts">
            {w.concepts.map((c) => (
              <div key={c.title} className="card">
                <h3>{c.title}</h3>
                <Paragraphs items={c.explain} />
              </div>
            ))}
          </div>

          <h2 id="changes">Every change, before and after</h2>
          <p>
            Left: the NumPy version (<code>main_functionized.py</code>). Right: the GPU version (<code>{CUDA}</code>). Everything not
            shown here is the same in both files.
          </p>
          {w.changes.map((c, i) => (
            <section key={i}>
              <h3 id={slug(c.title)}>
                <span className="section-num">{i + 1}</span>
                {c.title}
              </h3>
              <div className="rich">{c.why.map((p, k) => <p key={k}>{rich(p)}</p>)}</div>
              <div className="pair">
                <CodeBlock code={dedent(c.numpy)} name="NumPy" />
                <CodeBlock code={dedent(c.cuda)} name="CuPy (GPU)" />
              </div>
            </section>
          ))}

          <h2 id="trainer">The script that trained the model</h2>
          <Paragraphs items={w.trainer.intro} />
          {w.trainer.differences.map((d) => (
            <section key={d.title}>
              <h3 id={slug(d.title)}>{d.title}</h3>
              <Paragraphs items={d.explain} />
              {d.code && <CodeBlock code={dedent(d.code)} name={TRAINER} />}
            </section>
          ))}

          <h2 id="run">Run it yourself</h2>
          <p>
            Any computer with an NVIDIA GPU works, including a rented one. The model on this site trained on an RTX 2000 Ada from RunPod
            ($0.24 an hour) for 222 minutes, which came to $0.93. Without a GPU the same script still runs on NumPy, just much more slowly.
          </p>
          <CodeBlock code={RUN} lang="bash" name="terminal" />
          <p className="hint">
            If you rent a GPU, remember to stop or delete it when you&apos;re done: they charge by the hour whether they&apos;re working or not.
          </p>

          <h2 id="whole-files">The whole files</h2>
          <CodeBlock code={cuda} start={1} name={CUDA} tall download={`/code/${CUDA}`} />
          <CodeBlock code={trainer} start={1} name={TRAINER} tall download={`/code/${TRAINER}`} />
        </div>
      </div>
    </main>
  );
}
