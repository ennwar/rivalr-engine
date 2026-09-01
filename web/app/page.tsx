"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import LeaguePicker, { usePersistentIds } from "./components/LeaguePicker";

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

type ChipWar = {
  active_now?: string | null;
  chips_used: { name: string; event: number; live?: boolean }[];
  first_set_left: string[];
  expiry_gw: number;
  gws_to_expiry: number;
  bench_boost: { best_gw: number; swing: number } | null;
  triple_captain: { best_gw: number; swing: number; player: string } | null;
};

type Rival = {
  entry_id: number;
  name: string;
  team_name: string;
  rank: number | null;
  points: number;
  chips_left: Record<string, number>;
  chip_war?: ChipWar;
  overlap_pct: number | null;
  differentials: Player[];
  shields: Player[];
  swords: Player[];
};

const CHIP_LABEL: Record<string, string> = {
  wildcard: "WC",
  freehit: "FH",
  bboost: "BB",
  "3xc": "TC",
};

type Transfer = {
  out: Player | null;
  in: Player;
  net_gain: number;
  hits: number;
  swings: Record<string, number>;
  flags: string[];
};

type Action = { headline: string; why: string; do_nothing: string };

type Live = {
  gw: number;
  in_progress: boolean;
  my_players_to_play?: number;
  my_players_to_play_names?: string[];
};

type ModelStanding = {
  name: string;
  points: number;
  rank_in_league: number;
  through_gw: number;
};

type Brief = {
  action?: Action | null;
  model_team?: ModelStanding | null;
  live?: Live | null;
  free_transfers_now?: number | null;
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
  const [teamId, setTeamId, leagueId, setLeagueId] = usePersistentIds();
  const [target, setTarget] = useState<number | null>(null);
  const [brief, setBrief] = useState<Brief | null>(null);
  const [league, setLeague] = useState<LeagueEntry[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [firstTime, setFirstTime] = useState(false);
  const [notifyChatId, setNotifyChatId] = useState("");
  const [notifySent, setNotifySent] = useState(false);
  const runId = useRef(0);
  const lastQuery = useRef("");

  const countdown = useCountdown(brief?.deadline ?? null);

  const load = useCallback(
    async (targetOverride: number | null) => {
      const run = ++runId.current;
      setLoading(true);
      setError(null);
      setElapsed(0);
      setFirstTime(false);
      setNotifySent(false);
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
        lastQuery.current = q;

        // league (for my rank) in parallel; non-fatal
        fetch(`${API}/league?league_id=${encodeURIComponent(leagueId)}`)
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => run === runId.current && d && setLeague(d.entries))
          .catch(() => {});

        let r = await fetch(`${API}/brief?${q}`);
        if (r.status === 202) {
          const body202 = await r.json();
          const job_id = body202.job_id;
          if (body202.first_time && run === runId.current) setFirstTime(true);
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
        <LeaguePicker
          teamId={teamId}
          leagueId={leagueId}
          onTeamId={setTeamId}
          onLeagueId={setLeagueId}
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
            {firstTime
              ? `first time for this team/league - the full solve takes a few
                 minutes, after that it's instant`
              : `${elapsed}s elapsed · usually instant (pre-computed); a fresh
                 solve takes a few minutes`}
            {firstTime && ` · ${elapsed}s elapsed`}
          </div>
          <div className="bar">
            <div
              style={{ width: `${Math.min(95, (elapsed / 240) * 100)}%` }}
            />
          </div>
          {firstTime && !notifySent && (
            <div className="notifyrow">
              <input
                value={notifyChatId}
                onChange={(e) => setNotifyChatId(e.target.value.trim())}
                inputMode="numeric"
                placeholder="Telegram chat ID (optional)"
                aria-label="Telegram chat ID"
              />
              <button
                type="button"
                disabled={!notifyChatId}
                onClick={() => {
                  void fetch(
                    `${API}/brief?${lastQuery.current}&notify_chat_id=${encodeURIComponent(notifyChatId)}`,
                  ).catch(() => {});
                  setNotifySent(true);
                }}
              >
                ping me when ready
              </button>
              <div className="sub">
                don&apos;t wait around - message the rivalr bot on Telegram
                once (/start), drop your chat ID here, and close the tab
              </div>
            </div>
          )}
          {firstTime && notifySent && (
            <div className="sub">
              ✓ you&apos;ll get a Telegram message when it&apos;s ready - safe
              to close this tab
            </div>
          )}
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

          {brief.live?.in_progress && (
            <div className="warn" style={{ background: "#16233a", borderColor: "var(--accent)", color: "var(--text)" }}>
              GW{brief.live.gw} in progress
              {brief.live.my_players_to_play != null &&
                ` — ${brief.live.my_players_to_play} of your players still to play`}
              {brief.live.my_players_to_play_names?.length
                ? ` (${brief.live.my_players_to_play_names.join(", ")})`
                : ""}
              . Everything below plans for GW{brief.gameweek}.
            </div>
          )}

          {brief.action && (
            <section>
              <div className="hero">
                <div className="hero-label">this week</div>
                <div className="hero-headline">{brief.action.headline}</div>
                <div className="hero-why">{brief.action.why}</div>
                <div className="hero-nothing">
                  If you do nothing: {brief.action.do_nothing}
                </div>
                {brief.free_transfers_now != null && (
                  <div className="hero-ft mono">
                    {brief.free_transfers_now} free transfer
                    {brief.free_transfers_now !== 1 && "s"} available
                  </div>
                )}
              </div>
            </section>
          )}

          {brief.warnings.map((w) => (
            <div className="warn" key={w}>
              {w}
            </div>
          ))}

          {brief.rivals && brief.rivals.length > 0 && brief.rivals[0].chip_war && (
            <section>
              <h2>Chip war</h2>
              {brief.rivals[0].chip_war.gws_to_expiry > 0 &&
                brief.rivals[0].chip_war.gws_to_expiry <= 3 && (
                <div className="warn">
                  first chip set expires GW
                  {brief.rivals[0].chip_war.expiry_gw} —{" "}
                  {brief.rivals[0].chip_war.gws_to_expiry} gameweek
                  {brief.rivals[0].chip_war.gws_to_expiry !== 1 && "s"} left to
                  use or lose
                </div>
              )}
              {brief.rivals.map((r) => {
                const cw = r.chip_war!;
                return (
                  <div className="transfer" key={r.entry_id}>
                    <div className="line">
                      <span style={{ fontWeight: 600 }}>{r.name}</span>
                      {cw.active_now && (
                        <span
                          className="chip"
                          style={{ background: "var(--red)", color: "#fff" }}
                          title="active this gameweek - visible from their live picks before it reaches the history endpoint"
                        >
                          playing {CHIP_LABEL[cw.active_now] ?? cw.active_now} NOW
                        </span>
                      )}
                      <span style={{ marginLeft: "auto" }}>
                        {cw.first_set_left.map((c) => (
                          <span key={c} className="chip newclub" title={`${c} still unused from the first set (expires GW${cw.expiry_gw})`}>
                            {CHIP_LABEL[c] ?? c}
                          </span>
                        ))}
                        {cw.chips_used.map((c) => (
                          <span
                            key={`${c.name}${c.event}`}
                            className="chip"
                            style={{ textDecoration: "line-through", background: "#20242e", color: "var(--faint)" }}
                            title={`played ${c.name} in GW${c.event}`}
                          >
                            {CHIP_LABEL[c.name] ?? c.name}
                          </span>
                        ))}
                      </span>
                    </div>
                    <div className="swings">
                      {cw.bench_boost
                        ? `BB threat: +${cw.bench_boost.swing} best in GW${cw.bench_boost.best_gw}`
                        : "BB threat: unknown (squad hidden)"}
                      {" · "}
                      {cw.triple_captain
                        ? `TC threat: +${cw.triple_captain.swing} on ${cw.triple_captain.player} in GW${cw.triple_captain.best_gw}`
                        : "TC threat: unknown"}
                    </div>
                  </div>
                );
              })}
              <div className="legend">
                estimated from each rival&apos;s actual squad and our per-GW
                projections; doubles are already in the numbers
              </div>
            </section>
          )}

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
                {brief.model_team && (
                  <a href="/model" className="rival" style={{ borderColor: "var(--green)", textDecoration: "none" }}>
                    <div className="name" style={{ color: "var(--green)" }}>
                      🤖 {brief.model_team.name}
                    </div>
                    <div className="meta">
                      {brief.model_team.points} pts · rank{" "}
                      {brief.model_team.rank_in_league} · through GW
                      {brief.model_team.through_gw}
                    </div>
                    <div className="chips">
                      its own draft, its own transfers — tap for squads
                    </div>
                  </a>
                )}
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
                      <b>{r.points}</b> pts
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
                  <span className={`pos-badge pos-${p.position}`}>{p.position}</span>
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
                {(brief.captain as any).close_call && (
                  <div className="warn" style={{ marginTop: 8 }}>
                    The call is close: the top two are within 1.0 xPts —
                    either armband is defensible.
                  </div>
                )}
                {((brief.captain as any).board ?? []).length > 0 && (
                  <div className="fxwrap" style={{ marginTop: 10 }}>
                    <table className="fxtable" style={{ fontSize: 12.5 }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: "left" }}>candidate</th>
                          <th>xPts</th>
                          <th>fixture</th>
                          <th>opp recent PL form</th>
                          <th>margin</th>
                        </tr>
                      </thead>
                      <tbody>
                        {((brief.captain as any).board ?? []).map(
                          (c: any, i: number) => (
                            <tr key={c.id}>
                              <td style={{ textAlign: "left", fontWeight: i === 0 ? 700 : 400 }}>
                                {c.name}{i === 0 && " (C)"}
                              </td>
                              <td className="mono">{c.projection.toFixed(2)}</td>
                              <td>
                                {c.fixtures.length
                                  ? c.fixtures
                                      .map((f: any) => `${f.opponent} (${f.venue})`)
                                      .join(", ")
                                  : "blank"}
                              </td>
                              <td className="mono" style={{ whiteSpace: "nowrap" }}>
                                {c.fixtures
                                  .map((f: any) => {
                                    const o = f.opp_last5;
                                    if (!o) return "no data";
                                    const basis = o.thin
                                      ? `only ${o.matches} PL gms ⚠`
                                      : o.prev_season_matches > 0
                                        ? `${o.matches} gms · ${o.prev_season_matches} last szn`
                                        : `${o.matches} gms`;
                                    return `${o.xga_per_match} xGA · ${o.conceded} con · ${o.clean_sheets} CS (${basis})`;
                                  })
                                  .join(" / ") || "—"}
                              </td>
                              <td className="mono">
                                {c.margin_over_next != null
                                  ? `+${c.margin_over_next.toFixed(2)}`
                                  : ""}
                              </td>
                            </tr>
                          ),
                        )}
                      </tbody>
                    </table>
                    <div className="legend" style={{ marginTop: 8 }}>
                      {((brief.captain as any).board ?? [])
                        .filter((c: any) => c.blend)
                        .map((c: any) => {
                          const w = Math.round(c.blend.model_weight * 100);
                          return (
                            <div key={c.id}>
                              <b>{c.name}</b> {c.projection.toFixed(2)} ={" "}
                              {c.blend.model_raw.toFixed(2)} model +{" "}
                              {c.blend.ep_next.toFixed(1)} FPL, blended{" "}
                              {w}/{100 - w}
                              {w < 50 && " — mostly FPL's number"}
                            </div>
                          );
                        })}
                    </div>
                    <div className="legend">
                      margin = lead over the next candidate · opp recent PL
                      form = the opponent&apos;s actual defence (xGA per
                      match, goals conceded, clean sheets) over up to their
                      last 5 Premier League matches — real matches only,
                      nothing padded: promoted clubs have only this
                      season&apos;s games (⚠ = fewer than 5 exist), and
                      established clubs&apos; window can include last
                      May&apos;s fixtures, labelled &quot;last szn&quot; ·
                      blend = the
                      model&apos;s own raw output vs FPL&apos;s ep_next
                      before minutes/DefCon adjustments (early-season
                      cold-start; model share grows with matches played)
                    </div>
                  </div>
                )}
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

          <section>
            <h2>Ask rivalr</h2>
            <div className="notice">
              Questions about your week, your rivals, or the plan — answered
              from the solver and projections, in your own words too.{" "}
              <a href="/ask" style={{ color: "var(--accent)" }}>
                Open Ask rivalr →
              </a>
            </div>
          </section>

          <footer>
            projections: OpenFPL + DefCon layer · every recommendation is
            logged before the deadline and scored after -{" "}
            <a href="/accuracy">see our accuracy, including the failures</a>
          </footer>
        </>
      )}

      {!brief && !loading && !error && (
        <div>
          <div className="notice" style={{ fontSize: 14, color: "var(--text)" }}>
            rivalr shows what your transfers do to your position against the
            specific people in your mini-league — not the 10 million-player
            field. Projections, rival squads, chip threats, and one clear
            recommendation per week.
          </div>
          <div className="notice" style={{ marginTop: 8 }}>
            Find both IDs in any FPL URL: on fantasy.premierleague.com your
            points page looks like{" "}
            <span className="mono">…/entry/2616874/event/3</span> (team ID
            2616874) and your league table like{" "}
            <span className="mono">…/leagues/517089/standings/c</span> (league
            ID 517089).
          </div>
        </div>
      )}
    </main>
  );
}
