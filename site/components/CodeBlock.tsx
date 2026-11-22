import { highlight } from "@/lib/code";
import CopyButton from "./CopyButton";

type Props = {
  code: string;
  name?: string;          // shown in the bar above the code
  start?: number;         // the first line number (omit for no line numbers)
  lang?: string;
  tall?: boolean;         // cap the height and scroll inside (for whole files)
  download?: string;      // a url to download the whole file
};

export default async function CodeBlock({ code, name, start, lang = "python", tall, download }: Props) {
  const html = await highlight(code, lang);
  const lines = code.split("\n").length;
  const label = name ?? (start ? `lines ${start}–${start + lines - 1}` : "");
  return (
    <div className={"code" + (start ? "" : " nonum") + (tall ? " tall" : "")}>
      <div className="bar">
        <span className="name">{label}</span>
        {download && (
          <a className="copy" href={download} download style={{ textDecoration: "none" }}>
            Download
          </a>
        )}
        <CopyButton text={code} />
      </div>
      <div className="scroll" style={{ counterReset: `line ${(start ?? 1) - 1}` }} dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}
