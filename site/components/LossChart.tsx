"use client";
import { useState } from "react";

// from runpod/results/train.log: minutes into training, train loss, test loss (the run started at 10.15)
const DATA: [number, number, number][] = [
  [8.8, 2.621, 2.06], [17.7, 2.071, 1.92], [26.6, 1.958, 1.842], [35.5, 1.893, 1.786], [44.5, 1.853, 1.758],
  [53.4, 1.817, 1.726], [62.3, 1.79, 1.708], [71.2, 1.769, 1.699], [80.1, 1.748, 1.68], [89.0, 1.731, 1.675],
  [97.9, 1.714, 1.646], [106.8, 1.699, 1.629], [115.7, 1.685, 1.628], [124.6, 1.67, 1.612], [133.6, 1.658, 1.605],
  [142.5, 1.644, 1.591], [151.5, 1.632, 1.586], [160.4, 1.622, 1.582], [169.4, 1.614, 1.568], [178.3, 1.605, 1.56],
  [187.2, 1.596, 1.561], [196.2, 1.589, 1.553], [205.1, 1.586, 1.545], [214.1, 1.581, 1.546], [223.0, 1.575, 1.546],
];

const W = 760, H = 320, PAD = { l: 46, r: 64, t: 16, b: 36 };
const X0 = 0, X1 = 225, Y0 = 1.5, Y1 = 2.2;
const x = (m: number) => PAD.l + ((m - X0) / (X1 - X0)) * (W - PAD.l - PAD.r);
const y = (v: number) => PAD.t + (1 - (Math.min(v, Y1) - Y0) / (Y1 - Y0)) * (H - PAD.t - PAD.b);
const path = (i: 1 | 2) => DATA.map((d, k) => `${k ? "L" : "M"}${x(d[0]).toFixed(1)},${y(d[i]).toFixed(1)}`).join("");

export default function LossChart() {
  const [hover, setHover] = useState<number | null>(null);
  const h = hover !== null ? DATA[hover] : null;

  function move(e: React.PointerEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect();
    const mx = ((e.clientX - box.left) / box.width) * W;
    let best = 0;
    DATA.forEach((d, k) => { if (Math.abs(x(d[0]) - mx) < Math.abs(x(DATA[best][0]) - mx)) best = k; });
    setHover(best);
  }

  return (
    <figure className="card chart" style={{ margin: "1em 0" }}>
      <div className="legend">
        <span><i style={{ background: "var(--series-test)" }} /> test loss (stories it never trained on)</span>
        <span><i style={{ background: "var(--series-train)" }} /> train loss</span>
      </div>
      <div style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Test and train loss falling from about 2.1 to 1.55 over 223 minutes"
             onPointerMove={move} onPointerLeave={() => setHover(null)} style={{ display: "block", touchAction: "pan-y" }}>
          {[1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2].map((v) => (
            <g key={v}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} stroke="var(--line)" strokeWidth={1} />
              <text x={PAD.l - 8} y={y(v) + 4} textAnchor="end" className="axis">{v.toFixed(1)}</text>
            </g>
          ))}
          {[0, 30, 60, 90, 120, 150, 180, 210].map((m) => (
            <text key={m} x={x(m)} y={H - 12} textAnchor="middle" className="axis">{m} min</text>
          ))}
          <path d={path(2)} fill="none" stroke="var(--series-train)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          <path d={path(1)} fill="none" stroke="var(--series-test)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          <text x={x(223) + 6} y={y(1.575) + 4} className="dlabel" fill="var(--muted)">train</text>
          <text x={x(223) + 6} y={y(1.546) + 12} className="dlabel" fill="var(--muted)">test 1.545</text>
          {h && (
            <g>
              <line x1={x(h[0])} x2={x(h[0])} y1={PAD.t} y2={H - PAD.b} stroke="var(--muted)" strokeWidth={1} strokeDasharray="3 3" />
              <circle cx={x(h[0])} cy={y(h[1])} r={5} fill="var(--series-train)" stroke="var(--paper)" strokeWidth={2} />
              <circle cx={x(h[0])} cy={y(h[2])} r={5} fill="var(--series-test)" stroke="var(--paper)" strokeWidth={2} />
            </g>
          )}
        </svg>
        {h && (
          <div className="tip" style={{ left: `${(x(h[0]) / W) * 100}%` }}>
            <b>{h[0]} min</b>
            <span><i style={{ background: "var(--series-test)" }} />test {h[2].toFixed(3)}</span>
            <span><i style={{ background: "var(--series-train)" }} />train {h[1].toFixed(3)}</span>
          </div>
        )}
      </div>
      <figcaption className="hint" style={{ marginTop: 8 }}>
        The model started at 10.15 (random guessing) and was below 2.1 within 9 minutes, so the chart starts there. The test loss stayed
        under the train loss the whole time, which means it was learning how stories work, not memorizing them.
      </figcaption>
      <details className="hint" style={{ marginTop: 8 }}>
        <summary style={{ cursor: "pointer" }}>Show the numbers as a table</summary>
        <table className="data">
          <thead><tr><th>minutes</th><th>train loss</th><th>test loss</th></tr></thead>
          <tbody>{DATA.map((d) => <tr key={d[0]}><td>{d[0]}</td><td>{d[1].toFixed(3)}</td><td>{d[2].toFixed(3)}</td></tr>)}</tbody>
        </table>
      </details>
    </figure>
  );
}
