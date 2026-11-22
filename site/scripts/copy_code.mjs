// copies the python files from the project into public/code/, so the site shows (and lets people download) the real thing.
// a few casual comments get cleaned up on the way (only in the site's copies: the original files keep them).
// each one is swapped for a line with the same meaning, never deleted, so the walkthrough's line numbers still line up.
import fs from "node:fs";

const files = {
  "main_functionized.py": "../main_functionized.py",
  "main_functionized_cuda.py": "../main_functionized_cuda.py",
  "train_tinystories.py": "../runpod/train_tinystories.py",
};

const cleanups = [
  ["#it is vocab size bc it is all the tokens", "#it is vocab size because there is one row for every token"],
  ["#it is context length bc it embeds whatever it can remeber in the 64 token thingy yk",
   "#it is context length because there is one row for every position it can remember in the window"],
  ["#this legit just matrix multiplies the vectors", "#this just matrix multiplies the vectors"],
  ["#deadass just shifts around so qkv is the 4 heads", "#this just shifts things around so qkv is split into the heads"],
  ["#picking!!!!", "#the head: picking the next token"],
];

// anything that still looks like slang gets flagged, so a new casual comment doesn't slip onto the site unnoticed
const flagged = /\b(deadass|legit|thingy|yk|lol|bruh|ngl|lowkey|wtf|idk|fr|tbh)\b|!!!/i;

fs.mkdirSync("public/code", { recursive: true });
for (const [name, from] of Object.entries(files)) {
  // on vercel only this folder gets uploaded, so the originals aren't there: keep the copies already in public/code
  if (!fs.existsSync(from)) {
    if (!fs.existsSync("public/code/" + name)) throw new Error(`missing ${from} and public/code/${name}`);
    console.log(`  ${from} isn't here, using the copy already in public/code/`);
    continue;
  }
  let text = fs.readFileSync(from, "utf8");
  for (const [before, after] of cleanups) text = text.split(before).join(after);
  text.split("\n").forEach((line, i) => {
    if (line.trim().startsWith("#") && flagged.test(line)) console.warn(`  heads up: ${name} line ${i + 1} still has: ${line.trim()}`);
  });
  fs.writeFileSync("public/code/" + name, text);
}
console.log("copied", Object.keys(files).join(", "), "(with casual comments cleaned up)");
