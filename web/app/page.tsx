"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://rivalr-engine-production.up.railway.app";

// ---- types (mirror briefdata.py) ----------------------------------------

type Player = {
  id: number;
  name: string;
  club: string;
  position: string;
  price: number;
  projection: number;
  base: number;
  defcon: number;
  flags: string[];
  status: string | null;
  news: string;
  chance_of_playing: number | null;
};

type Rival = {
  entry_id: number;
  name: string;
  team_name: string;
  rank: number | null;
  points: number;
  chips_left: Record<string, number>;
  overlap_pct: number | null;
  differentials: Player[];
  shields: Player[];
  swords: Player[];
};

type Transfer = {
  out: Player | null;
  in: Player;
  net_gain: number;
  hits: number;
  swings: Record<string, number>;
  flags: string[];
};

type Brief = {
  gameweek: number;
  deadline: string;
  generated_at: string;
  mode: string;
  target_id: number | null;
  expected_points_horizon: number | null;
  squad: Player[];
  captain: { player: Player; reasoning: string } | null;
  transfers: Transfer[];
  rivals: Rival[] | null;
  warnings: string[];
  cached?: boolean;
};

type LeagueEntry = {
  entry_id: number;
  name: string;
  team_name: string;
  rank: number | null;
  points: number;
};

// ---- helpers -------------------------------------------------------------

const FLAG_INFO: Record<string, { cls: string; tip: string }> = {
  LOW_CONF: {
    cls: "lowconf",
    tip: "Low confidence: this projection sits at the model's floor for anyone who plays - it can't tell this pick apart from a blank.",
  },
  MGR_CHG: {
    cls: "mgrchg",
    tip: "Manager change: this club has a new boss, so the projection rests on last season's tactics until ~5 matches of new data.",
  },
  NEW_CLUB: {
    cls: "newclub",
    tip: "Transferred this summer: his stats were earned at a different club in a different system.",
  },
};

function Flags({ flags }: { flags: string[] }) {
  return (
    <>
      {flags.map((f) => {
        const info = FLAG_INFO[f];
        if (!info) return null;
        return (
          <span key={f} className={`chip ${info.cls}`} title={info.tip}>
            {f.replace("_", " ")}
          </span>
        );
      })}
    </>
  );
}

function useCountdown(deadline: string | null) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  if (!deadline) return null;
  const ms = new Date(deadline).getTime() - now;
  if (ms <= 0) return "deadline passed";
  const d = Math.floor(ms / 86400000);
  const h = Math.floor((ms % 86400000) / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  return d > 0 ? `${d}d ${h}h ${m}m` : `${h}h ${m}m ${s}s`;
}

// Honest, time-based stage estimate: the API doesn't expose solve stages,
// so these are labelled estimates, with real elapsed time shown.
const STAGES: [number, string][] = [
  [0, "fetching FPL + Understat data"],
  [15, "running the 200 OpenFPL models"],
  [55, "estimating minutes + DefCon corrections"],
  [75, "solving the transfer optimisation"],
];

function stageFor(elapsed: number): string {
  let s = STAGES[0][1];
  for (const [t, label] of STAGES) if (elapsed >= t) s = label;
  return s;
}

// ---- page ----------------------------------------------------------------

export default function Page() {
  const [teamId, setTeamId] = useState("2616874");
  const [leagueId, setLeagueId] = useState("517089");
  const [target, setTarget] = useState<number | null>(null);
  const [brief, setBrief] = useState<Brief | null>(null);
  const [league, setLeague] = useState<LeagueEntry[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const runId = useRef(0);

  const countdown = useCountdown(brief?.deadline ?? null);

  const load = useCallback(
    async (targetOverride: number | null) => {
      const run = ++runId.current;
      setLoading(true);
      setError(null);
      setElapsed(0);
      const t0 = Date.now();
      const tick = setInterval(
        () => setElapsed(Math.round((Date.now() - t0) / 1000)),
        1000,
      );
      try {
        const mode = targetOverride ? "chase" : "points";
        const q =
          `team_id=${encodeURIComponent(teamId)}` +
          `&league_id=${encodeURIComponent(leagueId)}&mode=${mode}` +
          (targetOverride ? `&target=${targetOverride}` : "");

        // league (for my rank) in parallel; non-fatal
        fetch(`${API}/league?league_id=${encodeURIComponent(leagueId)}`)
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => run === runId.current && d && setLeague(d.entries))
          .catch(() => {});

        let r = await fetch(`${API}/brief?${q}`);
        if (r.status === 202) {
          const { job_id } = await r.json();
          for (;;) {
            await new Promise((res) => setTimeout(res, 2500));
            if (run !== runId.current) return;
            const s = await fetch(`${API}/brief/status?job_id=${job_id}`);
            if (!s.ok) throw new Error(`status ${s.status}`);
            const body = await s.json();
            if (body.status === "done") {
              if (run === runId.current) setBrief(body.result);
              break;
            }
            if (body.status === "failed")
              throw new Error(body.error ?? "solve failed");
          }
        } else if (r.ok) {
          const body = await r.json();
          if (run === runId.current) setBrief(body);
        } else {
          const detail = await r.json().catch(() => null);
          throw new Error(detail?.detail ?? `API error ${r.status}`);
        }
      } catch (e) {
        if (run === runId.current)
          setError(e instanceof Error ? e.message : String(e));
      } finally {
        clearInterval(tick);
        if (run === runId.current) setLoading(false);
      }
    },
    [teamId, leagueId],
  );

  const pickTarget = (id: number) => {
    const next = target === id ? null : id;
    setTarget(next);
    void load(next);
  };

  const myRank =
    league?.find((e) => String(e.entry_id) === teamId)?.rank ?? null;

  return (
    <main>
      <form
        className="controls"
        onSubmit={(e) => {
          e.preventDefault();
          setTarget(null);
          void load(null);
        }}
      >
        <input
          value={teamId}
          onChange={(e) => setTeamId(e.target.value.trim())}
          inputMode="numeric"
          placeholder="FPL team ID"
          aria-label="FPL team ID"
        />
        <input
          value={leagueId}
          onChange={(e) => setLeagueId(e.target.value.trim())}
          inputMode="numeric"
          placeholder="mini-league ID"
          aria-label="mini-league ID"
        />
        <button disabled={loading || !teamId || !leagueId}>
          {loading ? "working…" : "get brief"}
        </button>
      </form>

      {error && (
        <div className="error">
          {/invalid|404|not found/i.test(error)
            ? `That team or league doesn't look right: ${error}`
            : error}
        </div>
      )}

      {loading && (
        <div className="progress">
          <div className="stage">{stageFor(elapsed)}…</div>
          <div className="sub">
            {elapsed}s elapsed · cold solve takes ~90s, cached results are
            instant · stage labels are estimates
          </div>
          <div className="bar">
            <div style={{ width: `${Math.min(95, (elapsed / 95) * 100)}%` }} />
          </div>
        </div>
      )}

      {brief && (
        <>
          <div className="gwbar">
            <h1>
              GW{brief.gameweek}
              {brief.cached && <span className="cached">cached ≤1h</span>}
            </h1>
            <span className="countdown mono">{countdown}</span>
            <span className="rank">
              {myRank ? `rank ${myRank} in league` : "league unranked (pre-season)"}
            </span>
          </div>

          {brief.warnings.map((w) => (
            <div className="warn" key={w}>
              {w}
            </div>
          ))}

          <section>
            <h2>Rivals{target ? " · chasing" : ""}</h2>
            {brief.rivals === null ? (
              <div className="notice">
                Rival intelligence unavailable right now - squad and transfers
                below are still valid.
              </div>
            ) : brief.rivals.length === 0 ? (
              <div className="notice">
                No rivals visible yet - mini-league squads appear after the
                first deadline.
              </div>
            ) : (
              <div className="rivals">
                {brief.rivals.map((r) => (
                  <button
                    key={r.entry_id}
                    className={`rival${target === r.entry_id ? " target" : ""}`}
                    onClick={() => pickTarget(r.entry_id)}
                    title={
                      target === r.entry_id
                        ? "click to clear target"
                        : "click to chase this rival"
                    }
                  >
                    <div className="name">{r.name}</div>
                    <div className="meta">
                      {r.points} pts
                      {r.overlap_pct != null && ` · ${r.overlap_pct}% overlap`}
                    </div>
                    <div className="chips">
                      chips:{" "}
                      {Object.entries(r.chips_left)
                        .filter(([, n]) => n > 0)
                        .map(([c, n]) => (n > 1 ? `${c}×${n}` : c))
                        .join(" ") || "none left"}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </section>

          <section>
            <h2>My squad</h2>
            {brief.squad.length === 0 ? (
              <div className="notice">
                Squad hidden (pre-season / picks not visible yet). The
                recommended draft is below.
              </div>
            ) : (
              brief.squad.map((p) => (
                <div className="rowline" key={p.id}>
                  <span className="pos">{p.position}</span>
                  <span className="pname">
                    {p.name}
                    <span className="club">{p.club}</span>
                    <Flags flags={p.flags} />
                  </span>
                  <span className="proj mono">
                    {p.projection.toFixed(2)}
                    {p.defcon >= 0.1 && (
                      <span className="def"> (+{p.defcon.toFixed(1)} def)</span>
                    )}
                  </span>
                </div>
              ))
            )}
            <div className="legend">
              <b>LOW CONF</b> projection at the model floor - can&apos;t be told
              apart from a blank · <b>MGR CHG</b> new manager, old-season
              tactics baked in · <b>NEW CLUB</b> stats earned in a different
              system
            </div>
          </section>

          {brief.captain && (
            <section>
              <h2>Captain</h2>
              <div className="captain">
                <div className="who">
                  {brief.captain.player.name}
                  <Flags flags={brief.captain.player.flags} />
                </div>
                <div className="why">{brief.captain.reasoning}</div>
              </div>
            </section>
          )}

          <section>
            <h2>
              Recommended transfers
              {brief.expected_points_horizon != null &&
                ` · ${brief.expected_points_horizon.toFixed(1)} xPts / 5 GW`}
            </h2>
            {brief.transfers.length === 0 ? (
              <div className="notice">Best move: bank the transfer.</div>
            ) : (
              brief.transfers.map((t, i) => (
                <div className="transfer" key={i}>
                  <div className="line">
                    {t.out && <span className="out">− {t.out.name}</span>}
                    <span className="in">
                      + {t.in.name}
                      <span className="club"> {t.in.club}</span>
                    </span>
                    <Flags flags={t.flags} />
                    <span className="gain mono">
                      {t.net_gain >= 0 ? "+" : ""}
                      {t.net_gain.toFixed(1)}
                    </span>
                  </div>
                  {Object.keys(t.swings).length > 0 && (
                    <div className="swings">
                      {Object.entries(t.swings)
                        .map(
                          ([who, v]) =>
                            `${v >= 0 ? "+" : ""}${v.toFixed(1)} on ${who}`,
                        )
                        .join(" · ")}
                    </div>
                  )}
                </div>
              ))
            )}
          </section>

          <footer>
            projections: OpenFPL + DefCon layer · every recommendation is
            logged before the deadline and scored after -{" "}
            <a
              href="https://github.com/ennwar/rivalr-engine/tree/main/logs/predictions"
              target="_blank"
              rel="noreferrer"
            >
              accuracy ledger
            </a>
          </footer>
        </>
      )}

      {!brief && !loading && !error && (
        <div className="notice">
          Enter your FPL team ID and mini-league ID, then get the brief.
        </div>
      )}
    </main>
  );
}
