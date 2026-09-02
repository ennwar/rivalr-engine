"use client";

import { useState } from "react";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://rivalr-engine-production.up.railway.app";

type Entity = {
  name: string;
  is_me: boolean;
  start_total: number;
  standstill: number[];
  optimal: number[];
  optimal_computed?: boolean;
};
type Traj = {
  gameweeks: number[];
  rivals_available: boolean;
  league_note?: string | null;
  rivals_capped?: boolean;
  entities: Entity[];
  honesty_note: string;
};

const SCENARIOS = [
  { id: 1, label: "I stand still, rivals stand still" },
  { id: 2, label: "I make the recommended moves, rivals stand still" },
  { id: 3, label: "I move, rivals also move optimally" },
];

// me: red; rivals: muted greys/greens
const ME_COLOR = "#4c9df3";
const RIVAL_COLORS = ["#e8b34b", "#3fd08c", "#a78bfa", "#5cc8e8", "#f06a6a",
  "#8494ab", "#6d7f99"];

export default function Trajectory({
  teamId, leagueId,
}: { teamId: string; leagueId: string }) {
  const [data, setData] = useState<Traj | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scenario, setScenario] = useState(2);

  const load = async () => {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const q = `team_id=${teamId}${leagueId ? `&league_id=${leagueId}` : ""}`;
      let r = await fetch(`${API}/trajectory?${q}`);
      if (r.status === 202) {
        const { job_id } = await r.json();
        for (;;) {
          await new Promise((res) => setTimeout(res, 3000));
          const s = await fetch(`${API}/brief/status?job_id=${job_id}`);
          const body = await s.json();
          if (body.status === "done") { setData(body.result); break; }
          if (body.status === "failed") throw new Error(body.error);
        }
      } else if (r.ok) {
        setData(await r.json());
      } else throw new Error(`API ${r.status}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // pick each entity's line for the current scenario
  const lineFor = (e: Entity): number[] =>
    (e.is_me ? scenario >= 2 : scenario === 3) ? e.optimal : e.standstill;

  return (
    <section>
      <h2>5-week trajectory — you vs the field</h2>
      <div className="notice">
        Cumulative projected points over the next 5 gameweeks, starting
        from everyone&apos;s current total. See the gap widen or close
        under different transfer scenarios.
      </div>

      {!data && (
        <div className="controls" style={{ marginTop: 8 }}>
          <button disabled={loading || !teamId} onClick={() => void load()}>
            {loading ? "projecting… (solves each team, ~1 min)" : "project 5 weeks"}
          </button>
        </div>
      )}
      {error && <div className="error">{error}</div>}

      {data && (() => {
        const gws = data.gameweeks;
        const hasRivals = data.rivals_available &&
          data.entities.some((e) => !e.is_me);
        const scenarios = hasRivals ? SCENARIOS : SCENARIOS.slice(0, 2);
        // x = [now, ...gws]; y = [start_total, ...cumulative]
        const xs = ["now", ...gws.map((g) => `GW${g}`)];
        const series = data.entities.map((e, i) => ({
          e,
          color: e.is_me ? ME_COLOR : RIVAL_COLORS[(i - 1) % RIVAL_COLORS.length],
          pts: [e.start_total, ...lineFor(e)],
          band: !e.is_me && scenario === 3 && e.optimal_computed
            ? { lo: [e.start_total, ...e.standstill], hi: [e.start_total, ...e.optimal] }
            : null,
        }));
        const allVals = series.flatMap((s) =>
          [...s.pts, ...(s.band ? [...s.band.lo, ...s.band.hi] : [])]);
        const maxY = Math.max(...allVals);
        const minY = Math.min(...allVals);
        const W = 640, H = 300, PAD = 34;
        const x = (i: number) => PAD + (i * (W - 2 * PAD)) / (xs.length - 1);
        const y = (v: number) =>
          H - PAD - ((v - minY) * (H - 2 * PAD)) / Math.max(maxY - minY, 1);
        const pathOf = (pts: number[]) =>
          pts.map((v, i) => `${x(i)},${y(v)}`).join(" ");

        return (
          <>
            <div className="controls" style={{ marginTop: 10, flexWrap: "wrap" }}>
              {scenarios.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  style={{
                    background: scenario === s.id ? "var(--accent)" : "var(--panel)",
                    color: scenario === s.id ? "#0a1524" : "var(--dim)",
                    border: "1px solid var(--border)", flex: "none",
                    fontSize: 12.5, textAlign: "left",
                  }}
                  onClick={() => setScenario(s.id)}
                >
                  {s.id}. {s.label}
                </button>
              ))}
            </div>

            <div className="warn" style={{ marginTop: 8 }}>
              {data.honesty_note}
            </div>

            <div className="chartwrap" style={{ marginTop: 10 }}>
              <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
                {[minY, (minY + maxY) / 2, maxY].map((v) => (
                  <g key={v}>
                    <line x1={PAD} x2={W - PAD} y1={y(v)} y2={y(v)}
                          stroke="#1a2130" strokeWidth="1" />
                    <text x={2} y={y(v) + 3} fontSize="10" fill="#5a6980">
                      {Math.round(v)}
                    </text>
                  </g>
                ))}
                {xs.map((lab, i) => (
                  <text key={lab} x={x(i)} y={H - 8} textAnchor="middle"
                        fontSize="10" fill="#5a6980">{lab}</text>
                ))}
                {/* rival optimal-uncertainty bands (scenario 3) */}
                {series.map((s) =>
                  s.band ? (
                    <polygon
                      key={`band-${s.e.name}`}
                      points={
                        s.band.hi.map((v, i) => `${x(i)},${y(v)}`).join(" ") + " " +
                        s.band.lo.map((v, i) => `${x(i)},${y(v)}`).reverse().join(" ")
                      }
                      fill={s.color} opacity="0.12"
                    />
                  ) : null,
                )}
                {series.map((s) => (
                  <polyline
                    key={s.e.name}
                    points={pathOf(s.pts)}
                    fill="none" stroke={s.color}
                    strokeWidth={s.e.is_me ? 3 : 2}
                    strokeDasharray={s.e.is_me ? undefined : "5 3"}
                  />
                ))}
              </svg>
              <div className="chartlegend">
                {series.map((s) => (
                  <span key={s.e.name}>
                    <span className="swatch" style={{ background: s.color }} />
                    {s.e.name}{s.e.is_me ? " (you)" : ""}
                  </span>
                ))}
              </div>
            </div>

            {data.rivals_capped && (
              <div className="notice">
                Only the first 8 rivals&apos; optimal plans are solved (cost);
                the rest use their stand-still line.
              </div>
            )}

            <div className="fxwrap" style={{ marginTop: 12 }}>
              <table className="fxtable" style={{ fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left" }}>manager</th>
                    <th>now</th>
                    {gws.map((g) => <th key={g}>GW{g}</th>)}
                    <th>vs you</th>
                  </tr>
                </thead>
                <tbody>
                  {series.map((s) => {
                    const meFinal =
                      series.find((x) => x.e.is_me)?.pts.slice(-1)[0] ?? 0;
                    const finalV = s.pts.slice(-1)[0];
                    const gap = s.e.is_me ? null : finalV - meFinal;
                    return (
                      <tr key={s.e.name}
                          style={{ fontWeight: s.e.is_me ? 700 : 400 }}>
                        <td style={{ textAlign: "left", color: s.color }}>
                          {s.e.name}{s.e.is_me ? " (you)" : ""}
                        </td>
                        <td className="mono">{Math.round(s.e.start_total)}</td>
                        {s.pts.slice(1).map((v, i) => (
                          <td className="mono" key={i}>{Math.round(v)}</td>
                        ))}
                        <td className="mono" style={{
                          color: gap == null ? undefined
                            : gap > 0 ? "var(--red)" : "var(--green)",
                        }}>
                          {gap == null ? "—"
                            : `${gap > 0 ? "+" : ""}${gap.toFixed(1)}`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="legend">
              &quot;vs you&quot; = each rival&apos;s projected 5-GW total minus
              yours (negative = you&apos;re ahead). Lines start at each
              manager&apos;s current total. Dashed = rivals, solid = you.
              {scenario === 3 &&
                " Shaded band = the range between a rival standing still and playing perfectly."}
            </div>
          </>
        );
      })()}
    </section>
  );
}
