"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Try it" },
  { href: "/numpy", label: "NumPy code" },
  { href: "/cuda", label: "GPU code" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <nav className="nav">
      <div className="wide inner">
        <Link href="/" className="brand">
          GPT <span>by hand</span>
        </Link>
        {links.map((l) => (
          <Link key={l.href} href={l.href} className={"link" + (path === l.href ? " on" : "")}>
            {l.label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
