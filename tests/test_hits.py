"""Hits toggle: the parameter must reach the solver as a binding
constraint, and plan-mode cache pairs must never be rebuilt as briefs."""

from rivalr import briefdata


def test_allow_hits_reaches_solver(monkeypatch):
    captured = {}

    def fake_solve(**kwargs):
        captured.update(kwargs.get("solver_options") or {})
        return {"points": None}  # short-circuit after capture

    monkeypatch.setattr(briefdata.optimise, "solve_all_modes", fake_solve)
    monkeypatch.setattr(briefdata.model, "project_all", lambda c, horizon: {})
    monkeypatch.setattr(briefdata.minutes, "estimate_minutes",
                        lambda c, p: None)
    monkeypatch.setattr(briefdata.minutes, "apply_minutes", lambda r, e: {})
    monkeypatch.setattr(
        briefdata.defcon, "DefConModel",
        lambda c: type("D", (), {"corrections": lambda *a, **k: {}})(),
    )

    class C:
        def bootstrap(self):
            return {"elements": [], "teams": [],
                    "events": [{"id": 3, "is_next": True,
                                "deadline_time": "2099-01-01T00:00:00Z"}]}

        def next_gw(self):
            return 3

    # no hits (default): both limits hard-zero
    try:
        briefdata.build_plan_json(C(), 1, 2, horizon=5)
    except RuntimeError:
        pass  # plan=None raises after capture - fine
    assert captured["weekly_hit_limit"] == 0
    assert captured["hit_limit"] == 0

    captured.clear()
    try:
        briefdata.build_plan_json(C(), 1, 2, horizon=5, allow_hits=True)
    except RuntimeError:
        pass
    assert captured["weekly_hit_limit"] == 2
    assert captured["hit_limit"] is None


def test_plan_modes_dispatch_to_plan_builder(monkeypatch):
    calls = []
    monkeypatch.setattr(
        briefdata, "build_plan_json",
        lambda c, t, l, horizon, allow_hits: calls.append(
            ("plan", horizon, allow_hits)) or {},
    )
    monkeypatch.setattr(
        briefdata, "build_brief_json",
        lambda c, t, l, mode, target_id: calls.append(("brief", mode)) or {},
    )
    briefdata.build_for_mode(None, 1, 2, "plan:h7:hits", None)
    briefdata.build_for_mode(None, 1, 2, "plan:h5", None)
    briefdata.build_for_mode(None, 1, 2, "points", None)
    assert calls == [("plan", 7, True), ("plan", 5, False), ("brief", "points")]
