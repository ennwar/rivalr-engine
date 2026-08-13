"""Vendored upstream repos: location, validation, sys.path wiring.

Both are cloned by scripts/setup_vendors.py into vendor/ (gitignored):

  vendor/OpenFPL                  daniegr/OpenFPL (trained models, inference only)
  vendor/FPL-Optimization-Tools   sertalpbilal/FPL-Optimization-Tools (HiGHS MILP)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger("rivalr.vendors")

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "vendor"
OPENFPL_DIR = VENDOR_DIR / "OpenFPL"
OPTIMIZER_DIR = VENDOR_DIR / "FPL-Optimization-Tools"

SETUP_HINT = "run `uv run python scripts/setup_vendors.py` to clone the upstream repos"


def require_openfpl() -> Path:
    if not (OPENFPL_DIR / "models").is_dir():
        log.error("OpenFPL vendor missing at %s - %s", OPENFPL_DIR, SETUP_HINT)
        raise FileNotFoundError(f"OpenFPL not vendored: {OPENFPL_DIR}. {SETUP_HINT}")
    return OPENFPL_DIR


def require_optimizer() -> Path:
    if not (OPTIMIZER_DIR / "dev" / "solver.py").is_file():
        log.error("FPL-Optimization-Tools vendor missing at %s - %s", OPTIMIZER_DIR, SETUP_HINT)
        raise FileNotFoundError(
            f"FPL-Optimization-Tools not vendored: {OPTIMIZER_DIR}. {SETUP_HINT}"
        )
    # Their modules use root-level imports (paths, utils, dev.*): repo root
    # must be on sys.path before `import dev.solver` works.
    root = str(OPTIMIZER_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    return OPTIMIZER_DIR
