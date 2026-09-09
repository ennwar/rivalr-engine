"""Player/team extraction for the widened Ask data layer."""

from rivalr import assistant as a

BOOT = {
    "teams": [
        {"id": 11, "name": "Liverpool", "short_name": "LIV"},
        {"id": 6, "name": "Chelsea", "short_name": "CHE"},
        {"id": 13, "name": "Man City", "short_name": "MCI"},
    ],
    "elements": [
        {"id": 367, "web_name": "Gakpo", "first_name": "Cody",
         "second_name": "Gakpo", "team": 11, "element_type": 3},
        {"id": 40, "web_name": "Rogers", "first_name": "Morgan",
         "second_name": "Rogers", "team": 6, "element_type": 3},
        {"id": 399, "web_name": "Cherki", "first_name": "Rayan",
         "second_name": "Cherki", "team": 13, "element_type": 3},
        {"id": 165, "web_name": "João Pedro", "first_name": "João",
         "second_name": "Pedro", "team": 6, "element_type": 4},
        {"id": 999, "web_name": "Best", "first_name": "George",
         "second_name": "Best", "team": 11, "element_type": 4},
    ],
}


def test_matches_named_players_across_teams():
    ids = a._match_players("how do Gakpo, Rogers and Cherki compare?", BOOT)
    assert set(ids) == {367, 40, 399}


def test_matches_accented_name_without_accent():
    ids = a._match_players("is joao pedro worth buying", BOOT)
    assert 165 in ids


def test_stopword_surname_not_falsely_matched():
    # "Best" is a real surname but also a common word - the question
    # uses it as a word, must NOT match the player.
    ids = a._match_players("who is the best captain this week", BOOT)
    assert 999 not in ids


def test_team_aliases_resolve():
    idx = a._team_index(BOOT)
    assert idx["liverpool"] == 11 and idx["liv"] == 11
    assert idx["chelsea"] == 6
    assert idx["city"] == 13 and idx["man city"] == 13
