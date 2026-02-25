#!/usr/bin/env bash
# =============================================================================
# run_pilot_mining.sh — Phase 1 parallel pilot runs
#
# Runs 5 factor-mining pilots simultaneously, each with a different
# REASONING_MODEL.  All share:
#   - CHAT_MODEL = qwen3-coder-next:cloud
#   - configs/experiment_pilot.yaml  (1 direction, 3 loops, no evolution)
#   - FACTOR_CoSTEER_FACTOR_ZOO_PATH pointing to zoo_v2.csv
#
# Usage:
#   bash scripts/run_pilot_mining.sh
#   bash scripts/run_pilot_mining.sh --dry-run    # print commands, don't exec
#
# Logs:  log/pilot_<suffix>/mining.log
# Libs:  data/factorlib/all_factors_library_pilot_<suffix>.json
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Load base env ────────────────────────────────────────────────────────────
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
else
    echo "❌  .env not found — copy configs/.env.example to .env first"; exit 1
fi

# ── Fixed shared config ──────────────────────────────────────────────────────
CHAT_MODEL_FIXED="qwen3-coder-next:cloud"
CONFIG_PATH="${REPO_ROOT}/configs/experiment_pilot.yaml"
ZOO_CSV="${REPO_ROOT}/data/factorlib/zoo_v2.csv"
DIRECTION="High-quality momentum and mean-reversion alpha factors for concentrated US equity portfolios with 15 assets. Focus on price-volume signals, volatility regimes, and risk-adjusted return patterns."
RESULTS_BASE="${DATA_RESULTS_DIR:-./data/results}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Build zoo_v2.csv if missing ──────────────────────────────────────────────
if [ ! -f "${ZOO_CSV}" ]; then
    echo "⚙️  Building zoo_v2.csv …"
    /home/hephaestus/anaconda3/envs/quantaalpha-ollama/bin/python3 \
        "${SCRIPT_DIR}/build_zoo_csv.py" --out "${ZOO_CSV}"
fi
echo "🦁  Zoo CSV: ${ZOO_CSV} ($(wc -l < "${ZOO_CSV}") rows)"

# ── Pilot definitions: SUFFIX → REASONING_MODEL ─────────────────────────────
declare -A PILOTS
PILOTS["pilot_glm5"]="glm-5:cloud"
PILOTS["pilot_minimax"]="minimax-m2.5:cloud"
PILOTS["pilot_qwen35"]="qwen3.5:397b"
PILOTS["pilot_nemotron"]="nemotron-3-nano:30b-cloud"
PILOTS["pilot_deepseek"]="deepseek-v3.2:cloud"

PIDS=()
LOG_DIR="${REPO_ROOT}/log"
mkdir -p "${LOG_DIR}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          QuantaAlpha — Phase 1 Pilot Mining                 ║"
echo "║  5 models × 1 direction × 3 loops  (parallel)              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

for SUFFIX in "${!PILOTS[@]}"; do
    REASONING="${PILOTS[$SUFFIX]}"
    PILOT_LOG_DIR="${LOG_DIR}/${SUFFIX}"
    PILOT_LOG="${PILOT_LOG_DIR}/mining.log"
    mkdir -p "${PILOT_LOG_DIR}"

    # Per-pilot isolated workspace & cache
    EXP_ID="${SUFFIX}_$(date +%Y%m%d_%H%M%S)"
    WS="${RESULTS_BASE}/workspace_${EXP_ID}"
    PC="${RESULTS_BASE}/pickle_cache_${EXP_ID}"

    CMD=(
        env
        LLM_CHAT_MODEL="${CHAT_MODEL_FIXED}"
        LLM_REASONING_MODEL="${REASONING}"
        LLM_OPENAI_BASE_URL="${LLM_OPENAI_BASE_URL:-https://ollama.com/v1}"
        LLM_OPENAI_API_KEY="${LLM_OPENAI_API_KEY:-}"
        LLM_LOG_LLM_CHAT_CONTENT=false
        FACTOR_LIBRARY_SUFFIX="${SUFFIX}"
        FACTOR_CoSTEER_FACTOR_ZOO_PATH="${ZOO_CSV}"
        EXPERIMENT_ID="${EXP_ID}"
        WORKSPACE_PATH="${WS}"
        PICKLE_CACHE_FOLDER_PATH_STR="${PC}"
        /home/hephaestus/anaconda3/envs/quantaalpha-ollama/bin/python3
            -m quantaalpha.cli mine
            --direction "${DIRECTION}"
            --config_path "${CONFIG_PATH}"
            --factor_lib_suffix "${SUFFIX}"
    )

    if $DRY_RUN; then
        echo "[DRY-RUN] ${SUFFIX}:"
        echo "  LLM_REASONING_MODEL=${REASONING}"
        echo "  log → ${PILOT_LOG}"
        echo ""
        continue
    fi

    echo "🚀  Launching ${SUFFIX} (${REASONING})"
    echo "    log → ${PILOT_LOG}"
    mkdir -p "${WS}" "${PC}"

    # Run in background, redirect stdout+stderr to log
    "${CMD[@]}" > "${PILOT_LOG}" 2>&1 &
    PIDS+=($!)
done

if $DRY_RUN; then
    echo "✅  Dry-run complete — no processes started."
    exit 0
fi

echo ""
echo "⏳  All 5 pilots running in background (PIDs: ${PIDS[*]})"
echo "    Monitor with:  tail -f log/pilot_<name>/mining.log"
echo ""

# ── Wait and collect results ─────────────────────────────────────────────────
echo "⏳  Waiting for all pilots to finish…"
FAILED=()
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAILED+=("${PID}")
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   Pilot Results Summary                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

FACTORLIB="${REPO_ROOT}/data/factorlib"
for SUFFIX in "${!PILOTS[@]}"; do
    LIB="${FACTORLIB}/all_factors_library_${SUFFIX}.json"
    if [ -f "${LIB}" ]; then
        STATS=$(/home/hephaestus/anaconda3/envs/quantaalpha-ollama/bin/python3 -c "
import json, pathlib
d = json.loads(pathlib.Path('${LIB}').read_text())
f = d.get('factors', {})
total = len(f)
hq = sum(1 for v in f.values() if v.get('quality') == 'high_quality')
mq = sum(1 for v in f.values() if v.get('quality') == 'medium_quality')
print(f'total={total} high={hq} medium={mq}')
" 2>/dev/null || echo "total=0 high=0 medium=0")
        echo "  ${SUFFIX} (${PILOTS[$SUFFIX]}): ${STATS}"
    else
        echo "  ${SUFFIX} (${PILOTS[$SUFFIX]}): ❌ no library found"
    fi
done

echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "⚠️  ${#FAILED[@]} pilot(s) exited with error (PIDs: ${FAILED[*]})"
    echo "    Check logs in log/pilot_*/mining.log"
else
    echo "✅  All pilots completed successfully."
fi

echo ""
echo "Next step — pick the best REASONING_MODEL and run Phase 2:"
echo "  bash scripts/run_full_mining.sh"
