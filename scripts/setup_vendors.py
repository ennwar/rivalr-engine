"""Clone the two upstream repos into vendor/ (shallow, pinned to main).

    uv run python scripts/setup_vendors.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "vendor"

REPOS = {
    "OpenFPL": "https://github.com/daniegr/OpenFPL",
    "FPL-Optimization-Tools": "https://github.com/sertalpbilal/FPL-Optimization-Tools",
}


def main() -> int:
    VENDOR.mkdir(exist_ok=True)
    for name, url in REPOS.items():
        dest = VENDOR / name
        if dest.exists():
            print(f"{name}: already present at {dest}, skipping")
            continue
        print(f"cloning {url} -> {dest}")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)], check=True
        )
    print("vendors ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
