"use client";

import { useCallback, useRef, useState } from "react";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://rivalr-engine-production.up.railway.app";

type Mini = {
  id: number;
  name: string;
  club: string;
  position: string;
  price: number;
  projection: number;
  still_to_play?: boolean | null;
};

type LiveState = { gw: number; in_progress: boolean };

type Week = {
  gw: number;
  transfers: { out: Mini | null; in: Mini }[];
  raw_gain?: number;
  hit_penalty?: number;
  net_gain?: number;
  banked: boolean;
  free_transfers: number | null;
  hits: number;
  itb: number | null;
  chip: string | null;
  captain: Mini | null;
  squad: Mini[];
  xp: number;
  cum_xp: number;
};

type Plan = {
  gameweek: number;
  horizon: number;
  live?: LiveState | null;
  locked: number[];
  banned: number[];
  free_transfers_now?: number | null;
  total_xp: number | null;
  weeks: Week[];
  cached?: boolean;
};

export default function Planner() {
  const [teamId, setTeamId] = useState("2616874");
  const [leagueId, setLeagueId] = useState("517089");
  const [horizon, setHorizon] = useState(5);
  const [allowHits, setAllowHits] = useState(false);
  const [locked, setLocked] = useState<Set<number>>(new Set());
  const [banned, setBanned] = useState<Set<number>>(new Set());
  const [plan, setPlan] = useState<Plan | null>(null);
  const [openWeek, setOpenWeek] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const runId = useRef(0);

  const load = useCallback(
    async (lockedSet: Set<number>, bannedSet: Set<number>, h: number, hitsOn: boolean) => {
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
        const q =
          `team_id=${encodeURIComponent(teamId)}` +
          `&league_id=${encodeURIComponent(leagueId)}&horizon=${h}` +
          (hitsOn ? `&hits=1` : "") +
          (lockedSet.size ? `&locked=${[...lockedSet].join(",")}` : "") +
          (bannedSet.size ? `&banned=${[...bannedSet].join(",")}` : "");
        let r = await fetch(`${API}/plan?${q}`);
        if (r.status === 202) {
          const { job_id } = await r.json();
          for (;;) {
            await new Promise((res) => setTimeout(res, 2500));
            if (run !== runId.current) return;
            const s = await fetch(`${API}/brief/status?job_id=${job_id}`);
            if (!s.ok) throw new Error(`status ${s.status}`);
            const body = await s.json();
            if (body.status === "done") {
              if (run === runId.current) setPlan(body.result);
              break;
            }
            if (body.status === "failed")
              throw new Error(body.error ?? "solve failed");
          }
        } else if (r.ok) {
          const body = await r.json();
          if (run === runId.current) setPlan(body);
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

  const toggle = (set: Set<number>, id: number): Set<number> => {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  };

  const lockIn = (id: number) => {
    const nextLocked = toggle(locked, id);
    const nextBanned = new Set(banned);
    nextBanned.delete(id);
    setLocked(nextLocked);
    setBanned(nextBanned);
  };
  const lockOut = (id: number) => {
    const nextBanned = toggle(banned, id);
    const nextLocked = new Set(locked);
    nextLocked.delete(id);
    setBanned(nextBanned);
    setLocked(nextLocked);
  };

  return (
    <main>
      <form
        className="controls"
        onSubmit={(e) => {
          e.preventDefault();
          void load(locked, banned, horizon, allowHits);
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
        <select
          value={horizon}
          onChange={(e) => setHorizon(Number(e.target.value))}
          aria-label="horizon"
        >
          {[1, 2, 3, 4, 5, 6, 7, 8].map((h) => (
            <option key={h} value={h}>
              {h} GW{h > 1 ? "s" : ""}
            </option>
          ))}
        </select>
        <button
          type="button"
          className={allowHits ? "" : ""}
          style={{
            background: allowHits ? "var(--amber)" : "var(--panel)",
            color: allowHits ? "#0a1524" : "var(--dim)",
            border: "1px solid var(--border)",
          }}
          title="off by default: the plan may not take -4 hits unless you allow it"
          onClick={() => setAllowHits(!allowHits)}
        >
          {allowHits ? "hits allowed" : "no hits"}
        </button>
        <button disabled={loading || !teamId || !leagueId}>
          {loading ? "solving…" : "get plan"}
        </button>
      </form>

      {(locked.size > 0 || banned.size > 0) && (
        <div className="warn">
          constraints: {locked.size > 0 && `${locked.size} locked in`}
          {locked.size > 0 && banned.size > 0 && " · "}
          {banned.size > 0 && `${banned.size} locked out`} ·{" "}
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault();
              setLocked(new Set());
              setBanned(new Set());
            }}
          >
            clear
          </a>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="progress">
          <div className="stage">solving the {horizon}-gameweek plan…</div>
          <div className="sub">
            {elapsed}s elapsed · longer horizons take longer; cached plans are
            instant
          </div>
          <div className="bar">
            <div style={{ width: `${Math.min(95, (elapsed / 180) * 100)}%` }} />
          </div>
        </div>
      )}

      {plan && (
        <>
          <div className="gwbar">
            <h1>
              plan GW{plan.gameweek}–{plan.gameweek + plan.horizon - 1}
              {plan.cached && <span className="cached">cached ≤1h</span>}
            </h1>
            <span className="rank mono">
              {plan.total_xp != null && `${plan.total_xp.toFixed(1)} xPts total`}
            </span>
          </div>
          {plan.live?.in_progress && (
            <div className="warn" style={{ background: "#16233a", borderColor: "var(--accent)", color: "var(--text)" }}>
              GW{plan.live.gw} is still being played. This plan starts at the
              GW{plan.gameweek} deadline — selling a player does NOT cost you
              his remaining GW{plan.live.gw} fixture. Consider waiting for
              GW{plan.live.gw} to finish before committing.
            </div>
          )}
          {plan.free_transfers_now != null && (
            <div className="warn" style={{ background: "#14251a", borderColor: "#1f4a2e", color: "var(--green)" }}>
              You have {plan.free_transfers_now} free transfer
              {plan.free_transfers_now !== 1 && "s"} going into GW
              {plan.gameweek} (reconstructed from your actual transfer
              history, banked FTs included).
            </div>
          )}

          {plan.weeks.map((w) => (
            <section key={w.gw}>
              <h2>
                GW{w.gw} · {w.xp.toFixed(1)} xPts · cum {w.cum_xp.toFixed(1)}
                {w.hits > 0 && ` · ${w.hits} hit${w.hits > 1 ? "s" : ""} (-${w.hits * 4})`}
                {w.chip && ` · ${w.chip}`}
              </h2>
              <div className="transfer">
                {w.banked ? (
                  <div className="line">
                    <span className="notice">
                      bank the free transfer
                      {w.free_transfers != null &&
                        ` (${w.free_transfers} FT after)`}
                    </span>
                  </div>
                ) : (
                  w.transfers.map((t, i) => (
                    <div className="line" key={i}>
                      {t.out && (
                        <span className="out">
                          − {t.out.name}{" "}
                          <span className="club">{t.out.club}</span>
                          {t.out.still_to_play && plan.live && (
                            <span className="chip lowconf"
                                  title={`still has a GW${plan.live.gw} fixture - his points there are yours no matter what; this sale only applies from GW${plan.gameweek}`}>
                              plays GW{plan.live.gw}
                            </span>
                          )}
                        </span>
                      )}
                      <span className="in">
                        + {t.in.name}{" "}
                        <span className="club">{t.in.club}</span>
                      </span>
                      <span className="gain mono">
                        {(t.in.projection - (t.out?.projection ?? 0)).toFixed(1)}
                      </span>
                    </div>
                  ))
                )}
                {w.hits > 0 && (
                  <div className="swings" style={{ color: "var(--amber)" }}>
                    hit justification: raw gain +{w.raw_gain?.toFixed(1)} over
                    the remaining horizon · penalty −{w.hit_penalty} · net{" "}
                    {w.net_gain != null && w.net_gain >= 0 ? "+" : ""}
                    {w.net_gain?.toFixed(1)}
                  </div>
                )}
                {w.itb != null && (
                  <div className="swings">
                    £{w.itb.toFixed(1)} in the bank
                    {w.captain && ` · captain ${w.captain.name}`}
                    {" · "}
                    <a
                      href="#"
                      onClick={(e) => {
                        e.preventDefault();
                        setOpenWeek(openWeek === w.gw ? null : w.gw);
                      }}
                    >
                      {openWeek === w.gw ? "hide squad" : "show squad"}
                    </a>
                  </div>
                )}
              </div>
              {openWeek === w.gw && (
                <div>
                  {w.squad.map((p) => (
                    <div className="rowline" key={p.id}>
                      <span className="pos">{p.position}</span>
                      <span className="pname">
                        {p.name} <span className="club">{p.club}</span>
                      </span>
                      <span className="proj mono">{p.projection.toFixed(2)}</span>
                      <button
                        type="button"
                        className={`chip ${locked.has(p.id) ? "lowconf" : ""}`}
                        title="lock IN: the plan must keep/buy this player"
                        onClick={() => lockIn(p.id)}
                      >
                        {locked.has(p.id) ? "locked" : "lock"}
                      </button>
                      <button
                        type="button"
                        className={`chip ${banned.has(p.id) ? "mgrchg" : ""}`}
                        title="lock OUT: the plan must never own this player"
                        onClick={() => lockOut(p.id)}
                      >
                        {banned.has(p.id) ? "banned" : "ban"}
                      </button>
                    </div>
                  ))}
                  <div className="legend">
                    toggle locks, then &quot;re-solve&quot; below
                  </div>
                </div>
              )}
            </section>
          ))}

          {(locked.size > 0 || banned.size > 0) && (
            <div className="controls" style={{ marginTop: 14 }}>
              <button
                disabled={loading}
                onClick={() => void load(locked, banned, horizon, allowHits)}
              >
                re-solve with {locked.size + banned.size} constraint
                {locked.size + banned.size !== 1 ? "s" : ""}
              </button>
            </div>
          )}
        </>
      )}

      {!plan && !loading && !error && (
        <div className="notice">
          The solver plans transfers across your chosen horizon: what to buy,
          when to bank, when a hit pays. Lock players in or out and it plans
          around them.
        </div>
      )}
    </main>
  );
}
