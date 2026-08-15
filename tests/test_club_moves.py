"""Club-membership resolution: live bootstrap is the single source of
truth for CURRENT club (fixtures, team features, flags); the 2025-26
prior supplies rates only, joined by permanent player code."""

from rivalr import uncertainty
from rivalr.defcon import DefConModel
from rivalr.minutes import MinutesEstimate


class MovedPlayerClient:
    """Player 77 (code 'C77') played 2025-26 at team 1 (old club) and is
    at team 2 (new club) in live bootstrap. Team 2 has a GW1 fixture,
    team 1 does not."""

    cache_dir = "data/cache"

    def bootstrap(self):
        return {
            "teams": [{"id": 1, "name": "Bournemouth"},
                      {"id": 2, "name": "Spurs"}],
            "elements": [{"id": 77, "code": "C77", "team": 2,
                          "element_type": 2, "web_name": "Mover"}],
            "events": [{"id": 1, "is_next": True,
                        "deadline_time": "2026-08-21T17:30:00Z"}],
        }

    def fixtures(self):
        return [
            {"event": 1, "finished": False, "team_h": 2, "team_a": 1},
        ]

    def element_summary(self, pid):
        return {"history": []}  # no 2026-27 matches yet

    def next_gw(self):
        return 1


def make_defcon(client):
    dc = DefConModel(client)
    # old-club prior rates keyed by player CODE, not club
    dc._prior_rates = {"C77": {"pos": "DEF", "rate90": 11.0,
                               "mins_avg": 90.0, "n": 30}}
    dc.group_fallback_rates = lambda: {"DEF": 5.0, "MIDFWD": 6.0}
    dc._params = {
        "features": [], "groups": {
            "DEF": {"coef": [0, 0, 0, 0, 0, 0], "intercept": 0.0},
            "MIDFWD": {"coef": [0, 0, 0, 0, 0, 0], "intercept": 0.0},
        },
    }
    return dc


def test_moved_player_keeps_old_club_rates():
    dc = make_defcon(MovedPlayerClient())
    el = MovedPlayerClient().bootstrap()["elements"][0]
    rate, mins = dc.blended_rate(el, "DEF")
    # zero 2026-27 matches: blend weight 0 -> pure old-club prior
    assert rate == 11.0 and mins == 90.0


def test_moved_player_gets_new_club_fixture():
    client = MovedPlayerClient()
    dc = make_defcon(client)
    est = {77: MinutesEstimate(77, p_start=0.9, expected_minutes=85.0,
                               factor=0.94, flags=[])}
    out = dc.corrections([77], est, horizon=1, team_hist={})
    # intercept-0 logistic -> P=0.5; nonzero correction proves the player
    # was matched to TEAM 2's GW1 fixture (his live club), not team 1's
    # absence of one.
    assert out[77][0] > 0.0


def test_transferred_flag_uses_live_club(monkeypatch, tmp_path):
    """NEW_CLUB flag: 25-26 club != live club, until 5 played matches."""
    import csv

    client = MovedPlayerClient()
    client.cache_dir = tmp_path
    with open(tmp_path / "vaastav_2025-26_players_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "team"])
        w.writeheader()
        w.writerow({"code": "C77", "team": 1})
    with open(tmp_path / "vaastav_2025-26_teams.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name"])
        w.writeheader()
        w.writerow({"id": 1, "name": "Bournemouth"})

    moved = uncertainty.transferred_players(client)
    assert moved[77]["from"] == "Bournemouth"
    assert moved[77]["to"] == "Spurs"

    flags = uncertainty.player_flags(client)
    assert "NEW_CLUB" in flags[77]["kinds"]
    # Spurs changed manager too: both kinds stack
    assert "MGR_CHG" in flags[77]["kinds"]
