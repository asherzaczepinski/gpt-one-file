import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Source_Serif_4 } from "next/font/google";
import Nav from "@/components/Nav";
import "./globals.css";

const serif = Source_Serif_4({ variable: "--font-serif", subsets: ["latin"] });
const sans = Inter({ variable: "--font-sans", subsets: ["latin"] });
const mono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "GPT by Hand", template: "%s · GPT by Hand" },
  description:
    "A GPT written by hand in NumPy by a high schooler, ported to the GPU with CuPy, trained for 93 cents. Try the model in your browser and read every line of the code.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable} ${mono.variable}`}>
      <body>
        <Nav />
        {children}
        <footer>
          <div className="wide">
            Built from scratch by a high schooler learning how GPTs work. The model runs entirely in your browser: nothing
            you type is sent anywhere.
          </div>
        </footer>
      </body>
    </html>
  );
}
