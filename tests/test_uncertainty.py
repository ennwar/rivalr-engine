"""Manager-change uncertainty flags: settle-after-5 and coverage."""

from rivalr import uncertainty
from rivalr.uncertainty import player_flags, team_flags


import csv


def make_prior_files(cache_dir):
    """Empty 2025-26 prior files: no transfers, no network."""
    with open(cache_dir / "vaastav_2025-26_players_raw.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["code", "team"]).writeheader()
    with open(cache_dir / "vaastav_2025-26_teams.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["id", "name"]).writeheader()


class StubClient:
    cache_dir = "."

    def element_summary(self, pid):
        return {"history": []}

    def __init__(self, finished_per_team: dict[int, int]):
        self._teams = [
            {"id": 1, "name": "Man City"},
            {"id": 2, "name": "Arsenal"},
            {"id": 3, "name": "Liverpool"},
        ]
        self._elements = [
            {"id": 10, "team": 1},
            {"id": 11, "team": 2},
            {"id": 12, "team": 3},
        ]
        # build fixtures giving each team the requested finished count
        self._fixtures = []
        for tid, n in finished_per_team.items():
            for _ in range(n):
                self._fixtures.append(
                    {"finished": True, "team_h": tid, "team_a": 99}
                )

    def bootstrap(self):
        return {"teams": self._teams, "elements": self._elements}

    def fixtures(self):
        return self._fixtures


def test_changed_club_flagged_until_five_matches():
    client = StubClient({1: 3, 3: 5})
    flags = team_flags(client)
    city = flags[1]
    assert city["active"] is True and city["matches_played"] == 3
    assert city["new"] == "Enzo Maresca" and city["out"] == "Pep Guardiola"
    liverpool = flags[3]
    assert liverpool["active"] is False  # 5 matches: settled


def test_unchanged_club_never_flagged():
    flags = team_flags(StubClient({2: 0}))
    assert 2 not in flags  # Arsenal: Arteta continuing


def test_player_flags_cover_changed_clubs_only(tmp_path):
    make_prior_files(tmp_path)
    client = StubClient({1: 0, 3: 5})
    client.cache_dir = tmp_path
    pf = player_flags(client)
    assert "MGR_CHG" in pf[10]["kinds"]   # City player, 0 matches -> flagged
    assert 11 not in pf                    # Arsenal player -> never
    assert 12 not in pf                    # Liverpool settled after 5


def test_all_config_teams_resolve_against_real_names():
    real = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
            'Chelsea', 'Coventry City', 'Crystal Palace', 'Everton', 'Fulham',
            'Hull City', 'Ipswich Town', 'Leeds', 'Liverpool', 'Man City',
            'Man Utd', 'Newcastle', "Nott'm Forest", 'Spurs', 'Sunderland']
    unknown = set(uncertainty.MANAGER_CHANGES) - set(real)
    assert not unknown, f"config names not in FPL bootstrap: {unknown}"
