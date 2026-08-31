"use client";

// Team-ID-first flow: enter your FPL team ID, we fetch your classic
// leagues from entry/{id}/ (system leagues like Overall/country already
// filtered server-side) and show them as a picker. Manual league-ID
// entry stays available as a fallback for leagues the API can't list.

import { useEffect, useRef, useState } from "react";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://rivalr-engine-production.up.railway.app";

const LS_TEAM = "rivalr.teamId";
const LS_LEAGUE = "rivalr.leagueId";

export function usePersistentIds(): [
  string, (v: string) => void, string, (v: string) => void,
] {
  const [teamId, setTeamId] = useState("2616874");
  const [leagueId, setLeagueId] = useState("517089");
  useEffect(() => {
    try {
      const t = localStorage.getItem(LS_TEAM);
      const l = localStorage.getItem(LS_LEAGUE);
      if (t) setTeamId(t);
      if (l) setLeagueId(l);
    } catch {}
  }, []);
  const saveTeam = (v: string) => {
    setTeamId(v);
    try { localStorage.setItem(LS_TEAM, v); } catch {}
  };
  const saveLeague = (v: string) => {
    setLeagueId(v);
    try { localStorage.setItem(LS_LEAGUE, v); } catch {}
  };
  return [teamId, saveTeam, leagueId, saveLeague];
}

type League = {
  league_id: number;
  name: string;
  entry_rank: number | null;
};

export default function LeaguePicker({
  teamId, leagueId, onTeamId, onLeagueId,
}: {
  teamId: string;
  leagueId: string;
  onTeamId: (v: string) => void;
  onLeagueId: (v: string) => void;
}) {
  const [leagues, setLeagues] = useState<League[] | null>(null);
  const [manual, setManual] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const leagueIdRef = useRef(leagueId);
  leagueIdRef.current = leagueId;

  useEffect(() => {
    if (timer.current) clearTimeout(timer.current);
    if (!/^\d{1,10}$/.test(teamId)) {
      setLeagues(null);
      setStatus(null);
      return;
    }
    timer.current = setTimeout(() => {
      setStatus("finding your leagues…");
      fetch(`${API}/myleagues?team_id=${teamId}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
        .then((d) => {
          setLeagues(d.leagues);
          if (!d.leagues.length) {
            setStatus("no mini-leagues found for that team - enter a league ID");
            setManual(true);
            return;
          }
          setStatus(null);
          // keep the current selection if it's one of theirs; else pick
          // the first mini-league so the form is immediately usable
          if (!d.leagues.some((l: League) => String(l.league_id) === leagueIdRef.current)) {
            onLeagueId(String(d.leagues[0].league_id));
          }
        })
        .catch(() => {
          setLeagues(null);
          setStatus("couldn't fetch your leagues - enter a league ID below");
          setManual(true);
        });
    }, 500);
    return () => { if (timer.current) clearTimeout(timer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teamId]);

  const showPicker = !manual && leagues && leagues.length > 0;

  return (
    <>
      <input
        value={teamId}
        onChange={(e) => onTeamId(e.target.value.trim())}
        inputMode="numeric"
        placeholder="FPL team ID"
        aria-label="FPL team ID"
      />
      {showPicker ? (
        <select
          value={leagueId}
          onChange={(e) => {
            if (e.target.value === "__manual__") setManual(true);
            else onLeagueId(e.target.value);
          }}
          aria-label="mini-league"
          style={{
            flex: 1, minWidth: 140, background: "var(--panel)",
            border: "1px solid var(--border)", color: "var(--text)",
            padding: "10px 12px", borderRadius: 8, fontSize: 15,
          }}
        >
          {leagues!.map((l) => (
            <option key={l.league_id} value={String(l.league_id)}>
              {l.name}{l.entry_rank != null ? ` (rank ${l.entry_rank})` : ""}
            </option>
          ))}
          <option value="__manual__">enter league ID manually…</option>
        </select>
      ) : (
        <input
          value={leagueId}
          onChange={(e) => onLeagueId(e.target.value.trim())}
          inputMode="numeric"
          placeholder="mini-league ID"
          aria-label="mini-league ID"
        />
      )}
      {manual && leagues && leagues.length > 0 && (
        <button
          type="button"
          style={{
            background: "var(--panel)", color: "var(--dim)",
            border: "1px solid var(--border)", flex: "none",
          }}
          onClick={() => setManual(false)}
          title="back to your leagues"
        >
          my leagues
        </button>
      )}
      {status && <div className="notice" style={{ flexBasis: "100%" }}>{status}</div>}
    </>
  );
}
