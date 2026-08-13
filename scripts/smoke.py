"""End-to-end smoke test on live data (no league/team required).

    uv run python scripts/smoke.py
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from rivalr.fetch import FPLClient
from rivalr.model import OpenFPLModel
from rivalr.minutes import estimate_minutes


def main() -> int:
    client = FPLClient()
    bootstrap = client.bootstrap()
    print(f"bootstrap OK: {len(bootstrap['elements'])} players, "
          f"{len(bootstrap['events'])} events")
    try:
        gw = client.next_gw()
        print(f"next GW: {gw}")
    except RuntimeError as exc:
        print(f"next_gw failed: {exc}")
        return 1

    # A small pool: the five most-owned players.
    pool = sorted(
        bootstrap["elements"],
        key=lambda el: -float(el["selected_by_percent"]),
    )[:5]
    pool_ids = [el["id"] for el in pool]
    print("pool:", [el["web_name"] for el in pool])

    m = OpenFPLModel(client)
    proj = m.project_all(horizon=3, pool=pool_ids)
    for el in pool:
        xs = proj.get(el["id"], [])
        e = estimate_minutes(client, el["id"])
        print(f"  {el['web_name']:<16} xPts {['%.2f' % x for x in xs]}"
              f"  xMins {e.expected_minutes} P(start) {e.p_start} {e.flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
