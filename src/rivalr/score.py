"""CLI: score a finished gameweek against the prediction ledger.

    python -m rivalr.score --gw 5
"""

from __future__ import annotations

import argparse
import logging

from .fetch import FPLClient
from .ledger import format_score_table, score_gw


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ledger predictions for a GW")
    parser.add_argument("--gw", type=int, required=True)
    parser.add_argument("--ledger-dir", default="logs/predictions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    client = FPLClient()
    result = score_gw(client, args.gw, ledger_dir=args.ledger_dir)
    print(format_score_table(result))


if __name__ == "__main__":
    main()
