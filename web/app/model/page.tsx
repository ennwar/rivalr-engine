"use client";

import { useEffect, useState } from "react";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://rivalr-engine-production.up.railway.app";

type Person = {
  entry_id: number;
  name: string;
  gameweeks: number[];
  points: number[];
  cum: number[];
};

type Season = {
  me: Person | null;
  model: { cum: number[]; edges: Record<string, number>; caveat: string } | null;
  model_team: {
    name: string;
    gameweeks: number[];
    points: number[];
    cum: number[];
    hits: number;
  } | null;
  rivals: Person[];
};

const RIVAL_COLORS = ["#8494ab", "#5a6980", "#6d7f99", "#4a5568"];

export default function ModelPage() {
  const [data, setData] = useState<Season | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [squads, setSquads] = useState<any | null>(null);
  const [squadGw, setSquadGw] = useState<number | null>(null);
  const [squadsLoading, setSquadsLoading] = useState(false);

  const loadSquads = (g: number) => {
    setSquadGw(g);
    setSquads(null);
    setSquadsLoading(true);
    fetch(`${API}/season/squads?team_id=2616874&league_id=517089&gw=${g}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setSquads(d))
      .finally(() => setSquadsLoading(false));
  };

  useEffect(() => {
    fetch(`${API}/season?team_id=2616874&league_id=517089`)
      .then((r) => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <main><div className="error">{error}</div></main>;
  if (!data) return <main><div className="notice">loading season…</div></main>;
  if (!data.me || data.me.gameweeks.length === 0)
    return (
      <main>
        <div className="notice">
          No scored gameweeks yet - this page fills in as the season runs.
        </div>
      </main>
    );

  const me = data.me;
  const gws = me.gameweeks;
  const series: { name: string; cum: number[]; color: string; dash?: boolean }[] = [
    { name: `me (${me.name.split(" ")[0]})`, cum: me.cum, color: "#4c9df3" },
  ];
  if (data.model_team)
    series.push({
      name: "🤖 The Model (autonomous)",
      cum: data.model_team.cum,
      color: "#3fd08c",
    });
  if (data.model)
    series.push({
      name: "you + the advice (secondary)",
      cum: data.model.cum,
      color: "#2a7a55",
      dash: true,
    });
  data.rivals.forEach((r, i) =>
    series.push({
      name: r.name.split(" ")[0],
      cum: r.cum.slice(0, gws.length),
      color: RIVAL_COLORS[i % RIVAL_COLORS.length],
    }),
  );

  const allVals = series.flatMap((s) => s.cum);
  const maxY = Math.max(...allVals, 1);
  const minY = Math.min(...allVals, 0);
  const W = 600;
  const H = 260;
  const PAD = 30;
  const x = (i: number) =>
    PAD + (i * (W - 2 * PAD)) / Math.max(gws.length - 1, 1);
  const y = (v: number) =>
    H - PAD - ((v - minY) * (H - 2 * PAD)) / Math.max(maxY - minY, 1);

  return (
    <main>
      <div className="gwbar">
        <h1>Me vs the model</h1>
      </div>
      <div className="notice">
        Is the tool actually helping, and are your friends beating it?
        Cumulative points: you, the you-that-followed-every-recommendation,
        and each rival.
      </div>

      <div className="chartwrap">
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
          {gws.map((g, i) => (
            <g key={g}>
              <text x={x(i)} y={H - 8} textAnchor="middle" fontSize="10"
                    fill="#5a6980">
                GW{g}
              </text>
            </g>
          ))}
          {[minY, Math.round((minY + maxY) / 2), maxY].map((v) => (
            <g key={v}>
              <line x1={PAD} x2={W - PAD} y1={y(v)} y2={y(v)}
                    stroke="#1a2130" strokeWidth="1" />
              <text x={4} y={y(v) + 3} fontSize="10" fill="#5a6980">{v}</text>
            </g>
          ))}
          {series.map((s) => (
            <polyline
              key={s.name}
              points={s.cum.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth={s.dash ? 2 : 2.5}
              strokeDasharray={s.dash ? "6 4" : undefined}
            />
          ))}
        </svg>
        <div className="chartlegend">
          {series.map((s) => (
            <span key={s.name}>
              <span className="swatch" style={{
                background: s.color,
                ...(s.dash ? { backgroundImage: "linear-gradient(90deg, transparent 0 25%, currentColor 25% 75%)" } : {}),
              }} />
              {s.name}
            </span>
          ))}
        </div>
      </div>

      <section>
        <h2>The squads (the evidence)</h2>
        <div className="notice">
          Tap a gameweek to see what the model owned vs what everyone
          actually owned - XI, captain, budget - so you can check it picks
          within a real budget.
        </div>
        <div className="controls" style={{ marginTop: 8 }}>
          {gws.map((g) => (
            <button
              key={g}
              type="button"
              style={{
                background: squadGw === g ? "var(--accent)" : "var(--panel)",
                color: squadGw === g ? "#0a1524" : "var(--dim)",
                border: "1px solid var(--border)", flex: "none",
              }}
              onClick={() => loadSquads(g)}
            >
              GW{g}
            </button>
          ))}
        </div>
        {squadsLoading && <div className="notice">loading squads…</div>}
        {squads && squadGw != null && (
          <div>
            {[
              { label: `me (${squads.me?.name ?? ""})`, s: squads.me },
              { label: "the model", s: squads.model },
              ...squads.rivals.map((r: any) => ({ label: r.name, s: r })),
            ].map(({ label, s }: any) =>
              s ? (
                <div className="transfer" key={label}>
                  <div className="line" style={{ fontWeight: 600 }}>
                    {label}
                    <span style={{ marginLeft: "auto" }} className="mono">
                      {s.gw_points != null && `${s.gw_points} pts · `}
                      {s.squad_value != null &&
                        `£${s.squad_value.toFixed(1)}m squad · £${(s.bank ?? 0).toFixed(1)}m ITB`}
                    </span>
                  </div>
                  {s.chip && (
                    <div className="swings">played {s.chip}</div>
                  )}
                  {s.note && <div className="swings">{s.note}</div>}
                  <div style={{ marginTop: 6 }}>
                    {s.players
                      .slice()
                      .sort((a: any, b: any) => b.points - a.points)
                      .map((p: any) => {
                        const mineIds = new Set(
                          (squads.me?.players ?? []).map((x: any) => x.id),
                        );
                        const divergent =
                          label !== `me (${squads.me?.name ?? ""})` &&
                          !mineIds.has(p.id);
                        return (
                          <span
                            key={p.id}
                            className="chip"
                            style={{
                              background: divergent ? "#2b2440" : "#20242e",
                              color: divergent ? "var(--violet)" : "var(--dim)",
                              margin: "2px 4px 2px 0",
                              fontSize: 11,
                            }}
                            title={`£${p.price_now.toFixed(1)}m now${p.recommended_in ? " · recommended in" : ""}`}
                          >
                            {p.name}
                            {p.captain && " (C)"} {p.points}
                          </span>
                        );
                      })}
                  </div>
                </div>
              ) : null,
            )}
            <div className="legend">
              purple = players you didn&apos;t own that gameweek · numbers are
              that GW&apos;s points · budget figures are real; player prices
              shown are current-day
            </div>
          </div>
        )}
      </section>

      <section>
        <h2>Per gameweek</h2>
        <div className="fxwrap">
          <table className="fxtable" style={{ fontSize: 13 }}>
            <thead>
              <tr>
                <th>GW</th>
                <th>my pts</th>
                <th>model edge</th>
                {data.rivals.map((r) => (
                  <th key={r.entry_id}>vs {r.name.split(" ")[0]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {gws.map((g, i) => (
                <tr key={g}>
                  <td>{g}</td>
                  <td className="mono">{me.points[i]}</td>
                  <td className="mono" style={{
                    color: (data.model?.edges[String(g)] ?? 0) > 0
                      ? "var(--green)"
                      : (data.model?.edges[String(g)] ?? 0) < 0
                        ? "var(--red)" : undefined,
                  }}>
                    {data.model?.edges[String(g)] != null
                      ? `${data.model.edges[String(g)] >= 0 ? "+" : ""}${data.model.edges[String(g)]}`
                      : "—"}
                  </td>
                  {data.rivals.map((r) => {
                    const d = me.points[i] - (r.points[i] ?? 0);
                    return (
                      <td key={r.entry_id} className="mono" style={{
                        color: d > 0 ? "var(--green)" : d < 0 ? "var(--red)" : undefined,
                      }}>
                        {d >= 0 ? "+" : ""}{d}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="legend">
          model edge = points the recommended transfers would have scored
          minus your actual transfers, per the pre-deadline ledger. Zero on
          unscored gameweeks. {data.model?.caveat}
        </div>
      </section>
    </main>
  );
}
