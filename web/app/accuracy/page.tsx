"use client";

import { useEffect, useState } from "react";

const API =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://rivalr-engine-production.up.railway.app";

type Bucket = { n: number; rmse: number | null; mae: number | null };
type Table = Record<string, Bucket>;

type Headline = {
  through_gw: number;
  model_points: number;
  model_rank: number;
  league_size: number;
  vs: { name: string; is_me: boolean; points: number; diff: number }[];
  captain_record: { model: number; human: number; tie: number };
  transfer_record: { gained: number; lost: number; net_points: number };
  hits_taken: number;
};

type Accuracy = {
  headline?: Headline | null;
  backtest: {
    reference: string;
    note: string;
    buckets: {
      bucket: string;
      n: number;
      ours: number;
      paper: number;
      benchmark: number;
      validated: boolean;
    }[];
  };
  live: {
    gw: number;
    partial_snapshot: boolean;
    accuracy: Table;
    accuracy_base: Table | null;
    unrostered: number;
  }[];
};

const BUCKETS = ["Zeros", "Blanks", "Tickers", "Haulers", "All"];

const LIMITATIONS: { title: string; body: string }[] = [
  {
    title: "The Zeros gap is an information problem, not a model problem",
    body: "Predicting who won't play at all depends on injury and team news. Our backtest had no historical availability flags, so it over-predicts non-players (+17% vs the paper). An oracle test with hindsight availability flips that bucket to -43%, which is why we attribute the gap to inputs rather than the pipeline. In-season we feed live FPL flags, but a player ruled out after our pre-deadline snapshot will still be a miss.",
  },
  {
    title: "The Blanks floor bias",
    body: "The model essentially never predicts below ~1.7 points for anyone who takes the pitch, so it systematically over-predicts players who play and do nothing (+1.1 points of bias on that bucket). We flag any recommendation resting on a near-floor projection as LOW CONFIDENCE rather than silently trusting it. Blanks accuracy is UNVALIDATED against the published benchmark and we say so.",
  },
  {
    title: "Expected minutes is our weakest component",
    body: "Who starts and for how long is estimated from rolling starts, availability flags and last season's minutes. It is purely statistical - no press conferences, no leaks. Early in the season it leans on last season's patterns, which is exactly when rotation surprises happen.",
  },
  {
    title: "Manager changes and transfers degrade the projections",
    body: "Ten clubs changed manager in summer 2026 and 41 players moved clubs. Their histories encode the old system. We flag these MGR CHG and NEW CLUB until roughly five matches of new evidence exist, but the underlying uncertainty is real and not fully correctable.",
  },
  {
    title: "DefCon is our own layer with its own error bars",
    body: "Defensive-contribution points did not exist when the base models were trained, so we add a calibrated correction (validated on 2025-26: DEF Brier 0.177 / AUC 0.698, MID-FWD Brier 0.115 / AUC 0.801, honest decile tables in the repo). The base-vs-corrected comparison below exists so the layer has to keep earning its place with live data.",
  },
];

function fmt(x: number | null | undefined): string {
  return x == null ? "—" : x.toFixed(3);
}

export default function AccuracyPage() {
  const [data, setData] = useState<Accuracy | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/accuracy`)
      .then((r) => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <main>
      <div className="gwbar">
        <h1>How accurate is this, honestly?</h1>
      </div>
      <div className="notice">
        Every recommendation is logged before the deadline and scored after
        the gameweek. The failures below are published on purpose: if you
        can&apos;t see where a model is weak, you can&apos;t trust where it
        claims to be strong.
      </div>

      {error && <div className="error">{error}</div>}

      {data && (
        <>
          {data.headline && (
            <section>
              <div className="hero">
                <div className="hero-label">the model, playing our league</div>
                <div className="hero-headline">
                  The Model is rank {data.headline.model_rank} of{" "}
                  {data.headline.league_size} with {data.headline.model_points}{" "}
                  points through GW{data.headline.through_gw}.
                </div>
                <div className="hero-why">
                  {data.headline.vs
                    .map(
                      (v) =>
                        `${v.is_me ? "me" : v.name.split(" ")[0]}: ${v.points} pts (model ${v.diff >= 0 ? "+" : ""}${v.diff})`,
                    )
                    .join(" · ")}
                </div>
                <div className="hero-nothing">
                  Captain pick: model beat mine{" "}
                  {data.headline.captain_record.model}×, mine beat it{" "}
                  {data.headline.captain_record.human}×, tied{" "}
                  {data.headline.captain_record.tie}× · Recommended transfers
                  gained points {data.headline.transfer_record.gained}× and
                  lost {data.headline.transfer_record.lost}× (net{" "}
                  {data.headline.transfer_record.net_points >= 0 ? "+" : ""}
                  {data.headline.transfer_record.net_points}) ·{" "}
                  {data.headline.hits_taken} hit(s) taken. An autonomous team:
                  its own draft, transfers and captains - it never sees ours.
                </div>
              </div>
            </section>
          )}

          <section>
            <h2>Backtest vs the published OpenFPL paper</h2>
            <div className="notice">{data.backtest.reference}</div>
            <div className="fxwrap">
              <table className="fxtable" style={{ fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left" }}>bucket</th>
                    <th>n</th>
                    <th>ours</th>
                    <th>benchmark</th>
                    <th>deviation</th>
                    <th>status</th>
                  </tr>
                </thead>
                <tbody>
                  {data.backtest.buckets.map((b) => {
                    const dev = (b.ours - b.benchmark) / b.benchmark;
                    return (
                      <tr key={b.bucket}>
                        <td style={{ textAlign: "left" }}>{b.bucket}</td>
                        <td className="mono">{b.n}</td>
                        <td className="mono">{b.ours.toFixed(3)}</td>
                        <td className="mono">
                          {b.benchmark.toFixed(3)}
                          {b.benchmark !== b.paper && (
                            <span className="cached"> (paper: {b.paper.toFixed(3)})</span>
                          )}
                        </td>
                        <td
                          className="mono"
                          style={{ color: Math.abs(dev) <= 0.1 ? "var(--green)" : "var(--amber)" }}
                        >
                          {(dev * 100).toFixed(1)}%
                        </td>
                        <td>{b.validated ? "validated" : "unvalidated"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="legend">{data.backtest.note}</div>
          </section>

          <section>
            <h2>Live season, gameweek by gameweek</h2>
            <div className="warn">
              A handful of gameweeks is noise, not proof - single-GW numbers
              swing wildly with how many players hauled that week. Judge
              trends from ~GW8 onward; we publish the early rows anyway
              because hiding them until they look good would defeat the
              point of this page.
            </div>
            {data.live.length === 0 ? (
              <div className="notice">
                No scored gameweeks yet - the first row appears once GW1
                finishes and settles. RMSE per bucket, same definitions as
                the paper: Zeros didn&apos;t play / Blanks ≤2 pts / Tickers
                3-4 / Haulers 5+.
              </div>
            ) : (
              <div className="fxwrap">
                <table className="fxtable" style={{ fontSize: 13 }}>
                  <thead>
                    <tr>
                      <th>GW</th>
                      {BUCKETS.map((b) => (
                        <th key={b}>{b}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.live.map((row) => (
                      <tr key={row.gw}>
                        <td>
                          {row.gw}
                          {row.partial_snapshot && (
                            <span className="cached" title="scored against a partial snapshot">*</span>
                          )}
                        </td>
                        {BUCKETS.map((b) => (
                          <td key={b} className="mono">
                            {fmt(row.accuracy[b]?.rmse)}
                            <span className="cached">n={row.accuracy[b]?.n ?? 0}</span>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section>
            <h2>Does the DefCon layer earn its place?</h2>
            {data.live.some((r) => r.accuracy_base) ? (
              <div className="fxwrap">
                <table className="fxtable" style={{ fontSize: 13 }}>
                  <thead>
                    <tr>
                      <th>GW</th>
                      {BUCKETS.map((b) => (
                        <th key={b}>{b} base → final</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.live
                      .filter((r) => r.accuracy_base)
                      .map((row) => (
                        <tr key={row.gw}>
                          <td>{row.gw}</td>
                          {BUCKETS.map((b) => {
                            const base = row.accuracy_base?.[b]?.rmse;
                            const fin = row.accuracy[b]?.rmse;
                            const better =
                              base != null && fin != null && fin < base;
                            return (
                              <td key={b} className="mono"
                                  style={{ color: better ? "var(--green)" : undefined }}>
                                {fmt(base)} → {fmt(fin)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="notice">
                Every snapshot logs the base projection and the
                DefCon-corrected one separately. Once gameweeks are scored,
                this table shows whether the correction reduced error - our
                stated expectation is that it lowers Blanks RMSE by moving
                defenders out of the ≤2-point bucket. If it doesn&apos;t, it
                gets re-examined.
              </div>
            )}
          </section>

          <section>
            <h2>Where we were wrong, and what we changed</h2>
            <div className="transfer">
              <div className="line" style={{ fontWeight: 600 }}>
                We retired FPL&apos;s ep_next from projections (GW3, 2026-09-02)
              </div>
              <div className="swings" style={{ marginTop: 6 }}>
                Early season we blended the model with FPL&apos;s own
                &quot;expected points&quot;. Then we measured it: across 357
                players, ep_next correlates 0.993 with the form stat —
                two gameweeks in it is literally a trailing average of
                two matches, with no fixture or venue content. Its
                track record here: GW1 it predicted 3.1 for both
                B.Fernandes and Haaland (both scored 2); GW2 it
                predicted 4.4 and 4.3 (they scored 23 and 13). Worse,
                it then chases whatever it missed — one 23-point haul
                pushed a +2.0 captain gap that our own model called a
                coin flip. ep_next now only covers GW1–2, when the
                model has no season data at all.
              </div>
            </div>
            <div className="transfer">
              <div className="line" style={{ fontWeight: 600 }}>
                We priced home advantage (GW3, 2026-09-02) — and it can
                be falsified
              </div>
              <div className="swings" style={{ marginTop: 6 }}>
                On 29,757 player-gameweeks of 2025-26, home is worth
                +0.375 points per match for attackers and +0.440 for
                defenders/keepers; our model priced it at ~zero. Now
                added (±half that, home vs away), recorded as its own
                ledger layer. The standing commitment, written before
                shipping: it must improve home-vs-away ordering
                accuracy over GW3–GW8 or it comes out — same deal as
                DefCon.
              </div>
            </div>
          </section>

          <section>
            <h2>Known limitations, in plain English</h2>
            {LIMITATIONS.map((l) => (
              <div className="transfer" key={l.title}>
                <div className="line" style={{ fontWeight: 600 }}>{l.title}</div>
                <div className="swings" style={{ marginTop: 6 }}>{l.body}</div>
              </div>
            ))}
          </section>

          <footer>
            raw data:{" "}
            <a
              href="https://github.com/ennwar/rivalr-engine/blob/main/docs/backtest_findings.md"
              target="_blank"
              rel="noreferrer"
            >
              full backtest findings
            </a>{" "}
            · every pre-deadline snapshot is committed to the repo before
            results are known
          </footer>
        </>
      )}
    </main>
  );
}
