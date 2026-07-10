/* Server-rendered SVG line chart (single series → no legend box; the title
   names it). Native <title> elements provide the per-point hover layer. */

type Point = { x: string; y: number };

export default function LineChart({
  points,
  height = 220,
  yLabel,
  baseline,
  baselineLabel,
}: {
  points: Point[];
  height?: number;
  yLabel: string;
  baseline?: number;
  baselineLabel?: string;
}) {
  if (points.length < 2) {
    return <div className="empty">Not enough data to chart yet.</div>;
  }

  const w = 720;
  const h = height;
  const pad = { l: 52, r: 16, t: 12, b: 26 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;

  const ys = points.map((p) => p.y).concat(baseline != null ? [baseline] : []);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  const span = yMax - yMin || 0.01;
  yMin -= span * 0.15;
  yMax += span * 0.15;

  const px = (i: number) => pad.l + (i / (points.length - 1)) * iw;
  const py = (v: number) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * ih;

  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${px(i).toFixed(1)},${py(p.y).toFixed(1)}`)
    .join(" ");

  const ticks = 4;
  const tickVals = Array.from(
    { length: ticks + 1 },
    (_, i) => yMin + ((yMax - yMin) * i) / ticks
  );

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      width="100%"
      role="img"
      aria-label={yLabel}
      style={{ display: "block" }}
    >
      {tickVals.map((v, i) => (
        <g key={i}>
          <line
            x1={pad.l}
            x2={w - pad.r}
            y1={py(v)}
            y2={py(v)}
            stroke="var(--grid)"
            strokeWidth="1"
          />
          <text
            x={pad.l - 8}
            y={py(v) + 4}
            textAnchor="end"
            fontSize="11"
            fill="var(--text-muted)"
          >
            {v.toFixed(3)}
          </text>
        </g>
      ))}
      {baseline != null && (
        <g>
          <line
            x1={pad.l}
            x2={w - pad.r}
            y1={py(baseline)}
            y2={py(baseline)}
            stroke="var(--baseline)"
            strokeWidth="1.5"
            strokeDasharray="5 4"
          />
          {baselineLabel && (
            <text
              x={w - pad.r}
              y={py(baseline) - 5}
              textAnchor="end"
              fontSize="11"
              fill="var(--text-muted)"
            >
              {baselineLabel}
            </text>
          )}
        </g>
      )}
      <path d={path} fill="none" stroke="var(--series-1)" strokeWidth="2" />
      {points.map((p, i) => (
        <circle
          key={i}
          cx={px(i)}
          cy={py(p.y)}
          r="4"
          fill="var(--series-1)"
          stroke="var(--surface-1)"
          strokeWidth="2"
        >
          <title>{`${p.x}: ${p.y.toFixed(4)}`}</title>
        </circle>
      ))}
      <text
        x={pad.l}
        y={h - 6}
        fontSize="11"
        fill="var(--text-muted)"
      >
        {points[0].x}
      </text>
      <text
        x={w - pad.r}
        y={h - 6}
        textAnchor="end"
        fontSize="11"
        fill="var(--text-muted)"
      >
        {points[points.length - 1].x}
      </text>
    </svg>
  );
}
