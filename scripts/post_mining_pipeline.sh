#!/usr/bin/env bash
# post_mining_pipeline.sh
#
# Run decay filter + combined backtest after a mining run completes.
#
# Usage (CN):
#   bash scripts/post_mining_pipeline.sh exp_phaseD
#
# Usage (US):
#   bash scripts/post_mining_pipeline.sh exp_us us
#
# Arguments:
#   $1  SUFFIX  — factor library suffix (e.g. exp_phaseD, exp_us)
#   $2  MARKET  — "cn" (default) or "us"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SCRIPT_DIR}/.."
cd "${REPO}"

PYTHON=/home/hephaestus/anaconda3/envs/quantaalpha-ollama/bin/python3

SUFFIX="${1:?Usage: $0 <suffix> [cn|us]}"
MARKET="${2:-cn}"

RAW_JSON="data/factorlib/all_factors_library_${SUFFIX}.json"
DECAY_JSON="data/factorlib/all_factors_library_${SUFFIX}_decay5d.json"

if [ "${MARKET}" = "us" ]; then
    HDF5="git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5"
    INSTRUMENTS_FILE="data/qlib/us_data_2025/instruments/sp500.txt"
    BACKTEST_CONFIG="configs/backtest_us_yahoo.yaml"
    OUTPUT_NAME="${SUFFIX}_combined_us"
    EXISTING_JSONS=()
else
    HDF5="git_ignore_folder/factor_implementation_source_data/daily_pv.h5"
    INSTRUMENTS_FILE="data/qlib/cn_data/instruments/csi300.txt"
    BACKTEST_CONFIG="configs/backtest.yaml"
    OUTPUT_NAME="${SUFFIX}_combined_topk30"
    EXISTING_JSONS=(
        "data/factorlib/all_factors_library_decay5d.json"
        "data/factorlib/all_factors_library_exp_phaseABC_decay5d.json"
        "data/factorlib/all_factors_library_exp_phaseABC2_decay5d.json"
    )
fi

echo "======================================================"
echo "Post-mining pipeline for: ${SUFFIX} (${MARKET})"
echo "======================================================"

# 1. Validate input library exists
if [ ! -f "${RAW_JSON}" ]; then
    echo "ERROR: Library not found: ${RAW_JSON}"
    echo "Mining may still be in progress. Check: data/factorlib/"
    exit 1
fi

RAW_COUNT=$($PYTHON -c "import json; d=json.load(open('${RAW_JSON}')); print(len(d.get('factors',{})))")
echo "Input library: ${RAW_JSON} (${RAW_COUNT} factors)"

# 2. Run decay filter
echo ""
echo "--- Step 1: Decay filter (5-day IC gate) ---"
$PYTHON scripts/filter_library_by_decay.py \
    --input "${RAW_JSON}" \
    --output "${DECAY_JSON}" \
    --horizon 5 \
    --min-ic 0.0 \
    --data-path "${HDF5}" \
    --instruments-file "${INSTRUMENTS_FILE}" \
    --use-abs-ic \
    --report

DECAY_COUNT=$($PYTHON -c "import json; d=json.load(open('${DECAY_JSON}')); print(len(d.get('factors',{})))")
echo "After decay filter: ${DECAY_COUNT} factors (dropped $((RAW_COUNT - DECAY_COUNT)))"

# 3. Run combined backtest
echo ""
echo "--- Step 2: Combined backtest (${BACKTEST_CONFIG}) ---"

FACTOR_JSON_ARGS="--factor-json ${DECAY_JSON}"
for f in "${EXISTING_JSONS[@]+"${EXISTING_JSONS[@]}"}"; do
    FACTOR_JSON_ARGS="${FACTOR_JSON_ARGS} --factor-json ${f}"
done

$PYTHON -m quantaalpha.backtest.run_backtest \
    -c "${BACKTEST_CONFIG}" \
    --factor-source combined \
    ${FACTOR_JSON_ARGS} \
    --output-name "${OUTPUT_NAME}" \
    -v

echo ""
echo "======================================================"
echo "Done! Results in: data/results/backtest_v2_results/${OUTPUT_NAME}_backtest_metrics.json"
echo "======================================================"
