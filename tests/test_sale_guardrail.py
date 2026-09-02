"""Sanity guardrail on counterintuitive sales (the Haaland->Gonzalo
class: MILP monetising a premium's price into budget elsewhere)."""

from rivalr.briefdata import _counterintuitive_sales, apply_sale_guardrail

ELS = {
    1: {"web_name": "Premium", "element_type": 4, "now_cost": 155},
    2: {"web_name": "Cheapie", "element_type": 4, "now_cost": 60},
    3: {"web_name": "Better", "element_type": 4, "now_cost": 80},
    4: {"web_name": "MidDef", "element_type": 2, "now_cost": 45},
}
FINAL = {
    1: [6.0] * 5,   # premium, 30 over horizon
    2: [4.8] * 5,   # 24 - does NOT beat the premium
    3: [6.5] * 5,   # 32.5 - clearly beats him
    4: [3.0] * 5,
}


def plan_with(outs, ins):
    return {"expected_points": 100.0,
            "weeks": [{"gw": 3, "transfers_in": ins, "transfers_out": outs}]}


def test_premium_sold_for_worse_replacement_is_flagged():
    bad = _counterintuitive_sales(ELS, FINAL, plan_with([1], [2]), 5)
    assert len(bad) == 1 and bad[0]["out_name"] == "Premium"


def test_premium_sold_for_clearly_better_is_fine():
    assert _counterintuitive_sales(ELS, FINAL, plan_with([1], [3]), 5) == []


def test_cheap_player_sale_never_flagged_by_price():
    # 4 is neither premium nor top-5 FWD (wrong position pool is tiny
    # here, so give him a non-top-5 projection universe)
    els = dict(ELS)
    fin = dict(FINAL)
    for i in range(10, 16):  # six better DEFs push him out of top-5
        els[i] = {"web_name": f"D{i}", "element_type": 2, "now_cost": 50}
        fin[i] = [5.0] * 5
    assert _counterintuitive_sales(els, fin, plan_with([4], [2]), 5) == []


def test_guardrail_serves_locked_resolve_and_warns():
    warnings = []
    locked_seen = {}

    def resolver(extra):
        locked_seen["ids"] = extra
        return {"expected_points": 97.5, "weeks": []}

    out = apply_sale_guardrail(
        plan_with([1], [2]), ELS, FINAL, 5, warnings, resolver)
    assert locked_seen["ids"] == [1]
    assert out["expected_points"] == 97.5
    assert any("kept them instead" in w for w in warnings)


def test_in_form_sale_warns_but_never_locks():
    """The Joao Pedro case: projections legitimately favour the swap,
    but the outgoing player is hot - warn, don't override the model."""
    els = {k: dict(v) for k, v in ELS.items()}
    els[4]["form"] = "10.0"   # in-form cheap player
    els[5] = {"web_name": "InDef", "element_type": 2, "now_cost": 55,
              "form": "4.0"}
    fin = dict(FINAL)
    fin[4] = [3.5] * 5        # sold for InDef (25 > 17.5: engine fine)
    fin[5] = [5.0] * 5
    for i in range(10, 16):   # better DEFs keep player 4 out of top-5
        els[i] = {"web_name": f"D{i}", "element_type": 2, "now_cost": 50}
        fin[i] = [6.0] * 5
    bad = _counterintuitive_sales(els, fin, plan_with([4], [5]), 5)
    assert len(bad) == 1 and bad[0]["action"] == "warn"

    warnings = []
    called = {}

    def resolver(ids):
        called["yes"] = True
        return None

    out = apply_sale_guardrail(
        plan_with([4], [5]), els, fin, 5, warnings, resolver)
    assert "yes" not in called          # no re-solve for form-only
    assert out["expected_points"] == 100.0
    assert any("rates short-term form conservatively" in w for w in warnings)


def test_guardrail_warns_loudly_when_resolve_fails():
    warnings = []
    out = apply_sale_guardrail(
        plan_with([1], [2]), ELS, FINAL, 5, warnings, lambda ids: None)
    assert out["expected_points"] == 100.0  # original kept
    assert any("SANITY CHECK" in w for w in warnings)
