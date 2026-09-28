# rivalr engine - web API (default CMD) and worker (override start command)
#
# Model artifacts (~750MB of OpenFPL joblib files) are NOT baked into the
# image: they're gitignored, and baking them in would add ~750MB to every
# deploy. The entrypoint clones both vendor repos into RIVALR_VENDOR_DIR
# on first boot - point that at a Railway volume so it happens once.

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- dependency layer -------------------------------------------------
# Keyed ONLY on the dependency manifests: a code-only change reuses this
# cached layer instead of re-installing the heavy ML stack
# (pandas/numpy/xgboost/scikit-learn/highspy) on every deploy - the fix
# that took code-change builds from minutes to seconds.
#
# This list MIRRORS pyproject.toml [project.dependencies] +
# [project.optional-dependencies].api, pins included. Keep it in sync
# with pyproject.toml: a missing/misversioned dep fails loudly at import
# (the entrypoint imports rivalr before serving), so drift can't ship
# silently. The package itself is installed --no-deps below.
COPY pyproject.toml uv.lock README.md ./
RUN pip install --no-cache-dir \
        "requests>=2.31" \
        "pandas>=2.3,<2.4" \
        "numpy>=1.26,<3" \
        "joblib==1.5.1" \
        "xgboost==3.0.2" \
        "scikit-learn==1.7.0" \
        "highspy>=1.11.0" \
        "fuzzywuzzy>=0.18" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "psycopg[binary]>=3.1" \
        "anthropic>=0.40"

# --- package layer (code only) ---------------------------------------
# --no-deps: the stack above is authoritative and already installed, so
# a change under src/ rebuilds ONLY this fast layer, not the ML stack.
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps .

# Persistent state lives on the mounted volume (default /data)
ENV RIVALR_VENDOR_DIR=/data/vendor \
    RIVALR_CACHE_DIR=/data/cache \
    RIVALR_LEDGER_DIR=/data/predictions \
    PYTHONUNBUFFERED=1

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

# Web process. Worker service overrides with: python -m rivalr.worker
CMD ["sh", "-c", "uvicorn rivalr.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
