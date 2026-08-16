"use client";

import { useEffect, useMemo, useState } from "react";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://rivalr-engine-production.up.railway.app";

type Fx = {
  opponent: string;
  home: boolean;
  fdr: number | null;
  our_def: number | null;
  our_att: number | null;
};

type Club = {
  team_id: number;
  name: string;
  short: string;
  fixtures: Record<string, Fx[]>;
};

type Grid = {
  gameweeks: number[];
  clubs: Club[];
  understat_available: boolean;
};

type Overlay = "fdr" | "our_def" | "our_att";

const OVERLAY_LABEL: Record<Overlay, string> = {
  fdr: "FPL FDR",
  our_def: "for defenders",
  our_att: "for attackers",
};

function diff(fx: Fx, overlay: Overlay): number | null {
  return fx[overlay] ?? fx.fdr ?? null;
}

export default function Fixtures() {
  const [grid, setGrid] = useState<Grid | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<Overlay>("fdr");
  const [from, setFrom] = useState<number | null>(null);
  const [to, setTo] = useState<number | null>(null);
  const [sorted, setSorted] = useState(false);

  useEffect(() => {
    fetch(`${API}/fixtures?horizon=8`)
      .then((r) => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      })
      .then((g: Grid) => {
        setGrid(g);
        setFrom(g.gameweeks[0]);
        setTo(g.gameweeks[g.gameweeks.length - 1]);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const avg = useMemo(() => {
    const out: Record<number, number> = {};
    if (!grid || from == null || to == null) return out;
    for (const c of grid.clubs) {
      let sum = 0;
      let n = 0;
      for (const gw of grid.gameweeks) {
        if (gw < from || gw > to) continue;
        const fxs = c.fixtures[String(gw)] ?? [];
        if (fxs.length === 0) {
          sum += 5; // a blank is as bad as the hardest fixture
          n += 1;
        }
        for (const fx of fxs) {
          const d = diff(fx, overlay);
          if (d != null) {
            sum += d;
            n += 1;
          }
        }
      }
      out[c.team_id] = n ? sum / n : 99;
    }
    return out;
  }, [grid, overlay, from, to]);

  const clubs = useMemo(() => {
    if (!grid) return [];
    const list = [...grid.clubs];
    if (sorted) list.sort((a, b) => (avg[a.team_id] ?? 99) - (avg[b.team_id] ?? 99));
    return list;
  }, [grid, sorted, avg]);

  if (error) return <main><div className="error">{error}</div></main>;
  if (!grid) return <main><div className="notice">loading fixtures…</div></main>;

  return (
    <main className="wide">
      <div className="fxcontrols">
        {(Object.keys(OVERLAY_LABEL) as Overlay[]).map((o) => (
          <button
            key={o}
            className={overlay === o ? "active" : ""}
            onClick={() => setOverlay(o)}
            disabled={o !== "fdr" && !grid.understat_available}
          >
            {OVERLAY_LABEL[o]}
          </button>
        ))}
        <span className="notice">avg over</span>
        <select value={from ?? ""} onChange={(e) => setFrom(Number(e.target.value))}>
          {grid.gameweeks.map((g) => (
            <option key={g} value={g}>GW{g}</option>
          ))}
        </select>
        <span className="notice">to</span>
        <select value={to ?? ""} onChange={(e) => setTo(Number(e.target.value))}>
          {grid.gameweeks.map((g) => (
            <option key={g} value={g}>GW{g}</option>
          ))}
        </select>
        <button className={sorted ? "active" : ""} onClick={() => setSorted(!sorted)}>
          {sorted ? "sorted easiest first" : "sort by avg"}
        </button>
      </div>
      {!grid.understat_available && (
        <div className="warn">
          Understat unavailable - our xG/xGA overlays are off, FPL FDR only.
        </div>
      )}
      <div className="fxwrap">
        <table className="fxtable">
          <thead>
            <tr>
              <th />
              {grid.gameweeks.map((g) => (
                <th key={g}>GW{g}</th>
              ))}
              <th>avg</th>
            </tr>
          </thead>
          <tbody>
            {clubs.map((c) => (
              <tr key={c.team_id}>
                <td className="club" title={c.name}>{c.short}</td>
                {grid.gameweeks.map((g) => {
                  const fxs = c.fixtures[String(g)] ?? [];
                  const inRange = from != null && to != null && g >= from && g <= to;
                  return (
                    <td key={g} style={{ opacity: inRange ? 1 : 0.35 }}>
                      {fxs.length === 0 && (
                        <span className="fxblank" title="blank gameweek">—</span>
                      )}
                      {fxs.length > 1 && (
                        <span className="fxdbl" title="double gameweek">DOUBLE</span>
                      )}
                      {fxs.map((fx, i) => {
                        const d = diff(fx, overlay);
                        return (
                          <div
                            key={i}
                            className={`fxcell ${d != null ? `fx${d}` : ""}`}
                            title={`${fx.home ? "home vs" : "away at"} ${fx.opponent}${
                              d != null ? ` · difficulty ${d}/5 (${OVERLAY_LABEL[overlay]})` : ""
                            }`}
                          >
                            {fx.home
                              ? fx.opponent.toUpperCase()
                              : fx.opponent.toLowerCase()}
                          </div>
                        );
                      })}
                    </td>
                  );
                })}
                <td className="fxavg mono">
                  {(avg[c.team_id] ?? 0) < 90 ? avg[c.team_id].toFixed(2) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="legend">
        UPPERCASE home, lowercase away · colours 1 easy → 5 hard ·
        &quot;for defenders&quot; = opponent&apos;s rolling xG ·
        &quot;for attackers&quot; = opponent&apos;s rolling xGA (inverted) ·
        blanks count as a 5 in averages
      </div>
    </main>
  );
}
