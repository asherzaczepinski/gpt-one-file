import type { Metadata } from "next";
import CodeBlock from "@/components/CodeBlock";
import CopyButton from "@/components/CopyButton";
import { Paragraphs } from "@/lib/rich";
import { lineRange, readCode } from "@/lib/code";
import walkthrough from "@/content/numpy-walkthrough.json";

export const metadata: Metadata = {
  title: "The NumPy version",
  description: "A whole GPT in one Python file with only NumPy: every section explained, every line ready to copy.",
};

const FILE = "main_functionized.py";
const SETUP = `pip install numpy tiktoken

# about 1 MB of short stories (the start of the TinyStories validation set)
curl -L https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt | head -c 1000000 > stories.txt

python ${FILE}`;

export default function NumpyPage() {
  const source = readCode(FILE);
  const total = source.split("\n").length;
  const sections = walkthrough.sections;

  return (
    <main className="wide">
      <section className="hero" style={{ maxWidth: 820 }}>
        <div className="eyebrow">The code · part 1</div>
        <h1>The NumPy version</h1>
        <Paragraphs items={walkthrough.intro} />
        <div className="row" style={{ marginTop: 10 }}>
          <a className="btn solid" href={`/code/${FILE}`} download>Download {FILE}</a>
          <CopyButton text={source} label="Copy the whole file" />
        </div>
      </section>

      <div className="page-grid" style={{ marginTop: 30 }}>
        <aside className="toc">
          <div className="t">On this page</div>
          <a href="#run">Run it yourself</a>
          {sections.map((s, i) => (
            <a key={s.id} href={`#${s.id}`}>{i + 1}. {s.title}</a>
          ))}
          <a href="#whole-file">The whole file</a>
        </aside>

        <div style={{ maxWidth: 820, minWidth: 0 }}>
          <h2 id="run" style={{ marginTop: 0 }}>Run it yourself</h2>
          <p>
            You need Python 3 with <code>numpy</code> and <code>tiktoken</code>, and a text file called <code>stories.txt</code> next to
            the script. Any plain text works. These commands grab the same kind of stories the model on the home page learned from:
          </p>
          <CodeBlock code={SETUP} lang="bash" name="terminal" />
          <p>
            It prints the loss as it trains (a few minutes on a laptop), saves the model to a <code>.pkl</code> file, and writes a few
            stories at the end.
          </p>

          {sections.map((s, i) => {
            const [start, end] = s.lines;
            const prevEnd = i === 0 ? 0 : sections[i - 1].lines[1];
            const gap = start - prevEnd - 1;
            return (
              <section key={s.id}>
                {gap >= 8 && (
                  <div className="gap-note">
                    {lineRange(source, prevEnd + 1, start - 1).trim()
                      ? `Lines ${prevEnd + 1}–${start - 1} are left out here: an earlier training loop (30 plain steps on one window), commented out and kept for reference. It's in the full file below.`
                      : `Lines ${prevEnd + 1}–${start - 1} are blank.`}
                  </div>
                )}
                <h2 id={s.id}>
                  <span className="section-num">{i + 1}</span>
                  {s.title}
                </h2>
                <Paragraphs items={s.explain} />
                <CodeBlock code={lineRange(source, start, end)} start={start} name={`${FILE} · lines ${start}–${end}`} />
              </section>
            );
          })}

          <h2 id="whole-file">The whole file</h2>
          <p>All {total} lines, exactly as they are, with the original comments.</p>
          <CodeBlock code={source} start={1} name={FILE} tall download={`/code/${FILE}`} />
        </div>
      </div>
    </main>
  );
}
