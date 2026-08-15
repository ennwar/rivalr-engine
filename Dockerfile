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

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --no-cache-dir . fastapi "uvicorn[standard]" "psycopg[binary]"

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
