#!/bin/sh
# First-boot setup: vendor repos onto the persistent volume, then exec.
set -e

mkdir -p "${RIVALR_VENDOR_DIR:-/data/vendor}" \
         "${RIVALR_CACHE_DIR:-/data/cache}" \
         "${RIVALR_LEDGER_DIR:-/data/predictions}"

if [ ! -d "${RIVALR_VENDOR_DIR}/OpenFPL/models" ]; then
    echo "entrypoint: vendoring OpenFPL (~750MB, first boot only)..."
    git clone --depth 1 https://github.com/daniegr/OpenFPL \
        "${RIVALR_VENDOR_DIR}/OpenFPL"
fi
if [ ! -f "${RIVALR_VENDOR_DIR}/FPL-Optimization-Tools/dev/solver.py" ]; then
    echo "entrypoint: vendoring FPL-Optimization-Tools..."
    git clone --depth 1 https://github.com/sertalpbilal/FPL-Optimization-Tools \
        "${RIVALR_VENDOR_DIR}/FPL-Optimization-Tools"
fi

exec "$@"
