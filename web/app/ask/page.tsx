"use client";

// Ask rivalr: suggested chips at the top, a conversation below, and
// free-text questions - all through the same grounding contract. The
// solver and projections make every decision; the LLM only ever sees
// engine JSON and may never invent a number. Unanswerable questions get
// a plain "the engine doesn't have data for that", never a guess.

import { useEffect, useRef, useState } from "react";

import LeaguePicker, { usePersistentIds } from "../components/LeaguePicker";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://rivalr-engine-production.up.railway.app";

type Turn = {
  question: string;
  answer: string;
  llm_used: boolean;
  data: any;
  showData?: boolean;
};

export default function AskPage() {
  const [teamId, setTeamId, leagueId, setLeagueId] = usePersistentIds();
  const [chips, setChips] = useState<{ id: string; label: string }[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [freeText, setFreeText] = useState("");
  const [loading, setLoading] = useState(false);
  const [pendingQ, setPendingQ] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!/^\d+$/.test(teamId)) return;  // league is optional
    fetch(`${API}/ask/questions?team_id=${teamId}${leagueId ? `&league_id=${leagueId}` : ""}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setChips(d.chips))
      .catch(() => {});
  }, [teamId, leagueId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, loading]);

  const ask = async (params: string, label: string) => {
    setLoading(true);
    setPendingQ(label);
    try {
      let result: any = null;
      const r = await fetch(
        `${API}/ask?team_id=${teamId}${leagueId ? `&league_id=${leagueId}` : ""}&${params}`,
      );
      if (r.status === 202) {
        const { job_id } = await r.json();
        for (;;) {
          await new Promise((res) => setTimeout(res, 2000));
          const s = await fetch(`${API}/brief/status?job_id=${job_id}`);
          const body = await s.json();
          if (body.status === "done") {
            result = body.result;
            break;
          }
          if (body.status === "failed") throw new Error(body.error);
        }
      } else if (r.ok) {
        result = await r.json();
      } else {
        throw new Error(`API ${r.status}`);
      }
      setTurns((t) => [
        ...t,
        {
          question: result.question || label,
          answer: result.answer,
          llm_used: !!result.llm_used,
          data: result.data ?? {},
        },
      ]);
    } catch (e) {
      setTurns((t) => [
        ...t,
        {
          question: label,
          answer: `Couldn't compute that right now (${e instanceof Error ? e.message : e}).`,
          llm_used: false,
          data: {},
        },
      ]);
    } finally {
      setLoading(false);
      setPendingQ(null);
    }
  };

  const askFree = () => {
    const q = freeText.trim();
    if (!q || loading) return;
    setFreeText("");
    void ask(`text=${encodeURIComponent(q)}`, q);
  };

  return (
    <main>
      <div className="controls">
        <LeaguePicker
          teamId={teamId}
          leagueId={leagueId}
          onTeamId={setTeamId}
          onLeagueId={setLeagueId}
        />
      </div>

      <div className="gwbar">
        <h1>Ask rivalr</h1>
      </div>
      <div className="notice">
        Answers come from the solver and projections — the AI only puts
        them into words. It never invents a number, and if the engine
        doesn&apos;t compute something, it says so.
      </div>

      <div className="askchips">
        {chips.map((c) => (
          <button
            key={c.id}
            type="button"
            disabled={loading}
            onClick={() => void ask(`qid=${c.id}`, c.label)}
          >
            {c.label}
          </button>
        ))}
      </div>

      <section>
        {turns.length === 0 && !loading && (
          <div className="notice">
            Pick a question above, or ask in your own words below.
          </div>
        )}
        {turns.map((t, i) => (
          <div className="askanswer" key={i} style={{ marginBottom: 10 }}>
            <div className="askq">{t.question}</div>
            <div className="aska">{t.answer}</div>
            <div className="asksrc">
              {t.llm_used
                ? "worded by AI from engine data only"
                : "engine data (readable summary unavailable)"}
              {" · "}
              <a
                href="#"
                onClick={(e) => {
                  e.preventDefault();
                  setTurns((ts) =>
                    ts.map((x, j) =>
                      j === i ? { ...x, showData: !x.showData } : x,
                    ),
                  );
                }}
              >
                {t.showData ? "hide" : "show"} the numbers
              </a>
            </div>
            {t.showData && (
              <pre className="askdata">{JSON.stringify(t.data, null, 1)}</pre>
            )}
          </div>
        ))}
        {loading && (
          <div className="askanswer" style={{ marginBottom: 10 }}>
            <div className="askq">{pendingQ}</div>
            <div className="aska" style={{ color: "var(--dim)" }}>
              computing from engine data…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </section>

      <form
        className="controls"
        style={{ marginTop: 14 }}
        onSubmit={(e) => {
          e.preventDefault();
          askFree();
        }}
      >
        <input
          value={freeText}
          onChange={(e) => setFreeText(e.target.value)}
          maxLength={300}
          placeholder="ask in your own words…"
          aria-label="free-text question"
        />
        <button disabled={loading || !freeText.trim()}>ask</button>
      </form>
      <div className="legend">
        Free-text answers are grounded in the cached brief (squad, plan,
        rivals, chips). Anything the engine doesn&apos;t compute — press
        conference news, other leagues, general football chat — it will
        decline rather than guess.
      </div>
    </main>
  );
}
