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


# -- matcher robustness (item 1) ------------------------------------------

ACCENT_BOOT = {
    "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"},
              {"id": 2, "name": "Leeds", "short_name": "LEE"}],
    "elements": [
        {"id": 15, "web_name": "Ødegaard", "first_name": "Martin",
         "second_name": "Ødegaard", "team": 1, "element_type": 3},
        {"id": 20, "web_name": "N.Williams", "first_name": "Neco",
         "second_name": "Williams", "team": 2, "element_type": 2},
        {"id": 21, "web_name": "M.Williams", "first_name": "Max",
         "second_name": "Williams", "team": 1, "element_type": 2},
    ],
}


def test_accented_name_resolves_without_accent():
    # the Ødegaard silent-drop bug: typing "Odegaard" must resolve
    assert a.resolve_players("will Odegaard start", ACCENT_BOOT)["resolved"] == [15]
    assert a.resolve_players("will Ødegaard start", ACCENT_BOOT)["resolved"] == [15]


def test_unknown_name_is_flagged_not_dropped():
    r = a.resolve_players("compare Messiah and Odegaard", ACCENT_BOOT)
    assert 15 in r["resolved"]
    assert "Messiah" in r["unresolved"]


def test_ambiguous_surname_lists_candidates():
    r = a.resolve_players("should I captain Williams", ACCENT_BOOT)
    assert r["resolved"] == []                      # never guesses
    assert set(r["ambiguous"].get("williams", [])) == {20, 21}


def test_dotted_shortform_resolves_uniquely():
    assert a.resolve_players("get N.Williams", ACCENT_BOOT)["resolved"] == [20]
