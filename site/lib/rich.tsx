import { Fragment, type ReactNode } from "react";

// the walkthrough text uses `code` and **bold**. this turns those into elements (everything else stays plain text)
export function rich(text: string): ReactNode {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((part, i) => {
    if (part.startsWith("`") && part.endsWith("`")) return <code key={i}>{part.slice(1, -1)}</code>;
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    return <Fragment key={i}>{part}</Fragment>;
  });
}

export function Paragraphs({ items }: { items: string[] }) {
  return (
    <div className="rich">
      {items.map((p, i) => (
        <p key={i}>{rich(p)}</p>
      ))}
    </div>
  );
}
