#!/usr/bin/env bash
# =============================================================================
# run_evolution_mining.sh — Evolution-mode mining (original → mutation → crossover)
#
# Runs 5 miners in parallel, each with a different reasoning model.
# Uses zoo_v2_extended.csv (pilot factors included) to avoid redundant discovery.
#
# Usage:
#   bash scripts/run_evolution_mining.sh
#   bash scripts/run_evolution_mining.sh --dry-run
#
# Logs:  log/evo_<suffix>/run_<timestamp>.log
# Libs:  data/factorlib/all_factors_library_evo_<suffix>.json
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Load base env ────────────────────────────────────────────────────────────
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
else
    echo "❌  .env not found"; exit 1
fi

TIMESTAMP=$(date +%H%M%S)
CONFIG_PATH="${REPO_ROOT}/configs/experiment_evolution.yaml"
ZOO_CSV="${REPO_ROOT}/data/factorlib/zoo_v2_extended.csv"
DIRECTION="High-quality momentum and mean-reversion alpha factors for concentrated US equity portfolios with 15 assets. Focus on price-volume signals, volatility regimes, and risk-adjusted return patterns."

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

echo "🧬  Evolution run — timestamp: ${TIMESTAMP}"
echo "🦁  Zoo: ${ZOO_CSV} ($(wc -l < "${ZOO_CSV}") rows)"

# ── Pilot definitions: SUFFIX → REASONING_MODEL ─────────────────────────────
declare -A PILOTS
PILOTS["evo_deepseek"]="deepseek-v3.2:cloud"
PILOTS["evo_glm5"]="glm-5:cloud"
PILOTS["evo_minimax"]="minimax-m2.5:cloud"
PILOTS["evo_nemotron"]="nemotron-3-nano:30b-cloud"
PILOTS["evo_qwen35"]="qwen3.5:397b"

PIDS=()

for SUFFIX in "${!PILOTS[@]}"; do
    REASON="${PILOTS[$SUFFIX]}"
    LOG_DIR="${REPO_ROOT}/log/${SUFFIX}"
    mkdir -p "${LOG_DIR}"
    LOG="${LOG_DIR}/run_${TIMESTAMP}.log"

    EXP_ID="${SUFFIX}_$(date +%Y%m%d_%H%M%S)"
    WS="${REPO_ROOT}/workspace/${SUFFIX}_${TIMESTAMP}"
    CACHE="${REPO_ROOT}/pickle_cache/${SUFFIX}_${TIMESTAMP}"

    CMD=(
        /home/hephaestus/anaconda3/envs/quantaalpha-ollama/bin/python3
        -m quantaalpha.cli mine
        --direction "${DIRECTION}"
        --config_path "${CONFIG_PATH}"
    )

    ENV_VARS=(
        "EXPERIMENT_ID=${EXP_ID}"
        "WORKSPACE_PATH=${WS}"
        "PICKLE_CACHE_FOLDER_PATH_STR=${CACHE}"
        "LLM_CHAT_MODEL=qwen3-coder-next:cloud"
        "LLM_REASONING_MODEL=${REASON}"
        "FACTOR_LIBRARY_SUFFIX=${SUFFIX}"
        "FACTOR_CoSTEER_FACTOR_ZOO_PATH=${ZOO_CSV}"
        "LLM_CHAT_MAX_TOKENS=32000"
    )

    if $DRY_RUN; then
        echo "  [DRY] ${SUFFIX}: ${ENV_VARS[*]} ${CMD[*]}"
        continue
    fi

    env "${ENV_VARS[@]}" nohup "${CMD[@]}" > "${LOG}" 2>&1 &
    PID=$!
    PIDS+=("$PID")
    echo "  ✅  ${SUFFIX} PID=${PID}  REASON=${REASON}  LOG=${LOG}"
    sleep 1
done

if ! $DRY_RUN; then
    echo ""
    echo "🚀  All ${#PIDS[@]} evolution runners launched."
    echo "    Monitor: tail -f log/evo_*/run_${TIMESTAMP}.log"
    echo "    Results: data/factorlib/all_factors_library_evo_*.json"
fi
