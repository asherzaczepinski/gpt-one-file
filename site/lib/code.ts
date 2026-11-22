import fs from "node:fs";
import path from "node:path";
import { createHighlighter, type Highlighter } from "shiki";

// the python files, as copied into public/code by scripts/copy_code.mjs
export function readCode(name: string): string {
  return fs.readFileSync(path.join(process.cwd(), "public", "code", name), "utf8");
}

// lines start..end (1-indexed, inclusive) of a file
export function lineRange(source: string, start: number, end: number): string {
  return source.split("\n").slice(start - 1, end).join("\n");
}

// remove the indentation every line shares, so a snippet from inside a function starts at the left edge
export function dedent(text: string): string {
  const lines = text.split("\n");
  const indents = lines.filter((l) => l.trim()).map((l) => l.match(/^ */)![0].length);
  const cut = indents.length ? Math.min(...indents) : 0;
  return lines.map((l) => l.slice(cut)).join("\n");
}

let highlighter: Promise<Highlighter> | null = null;
export async function highlight(code: string, lang = "python"): Promise<string> {
  highlighter ??= createHighlighter({ themes: ["github-light", "github-dark"], langs: ["python", "bash"] });
  return (await highlighter).codeToHtml(code, {
    lang,
    themes: { light: "github-light", dark: "github-dark" },
    defaultColor: false,
  });
}
