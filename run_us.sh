#!/bin/bash
# QuantaAlpha US (S&P500) experiment runner
#
# Usage:
#   ./run_us.sh "initial direction"
#   ./run_us.sh "initial direction" "suffix"
#
# Examples:
#   ./run_us.sh "S&P500 Price-Volume Factor Mining"
#   ./run_us.sh "S&P500 Momentum Factors" "exp_us_momentum"
#
# Prerequisites:
#   - data/qlib/us_data_2025/   (calendars, features, instruments)
#   - git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Load base .env
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
else
    echo "Error: .env file not found"
    exit 1
fi

# =============================================================================
# US-specific overrides
# =============================================================================
US_QLIB_DATA="${SCRIPT_DIR}/data/qlib/us_data_2025"
US_HDF5_DIR="${SCRIPT_DIR}/git_ignore_folder/factor_implementation_source_data_us"
US_HDF5_DEBUG_DIR="${SCRIPT_DIR}/git_ignore_folder/factor_implementation_source_data_us_debug"

# Validate US data
if [ ! -d "${US_QLIB_DATA}/instruments" ]; then
    echo "Error: US Qlib data not found at ${US_QLIB_DATA}"
    echo "Expected: data/qlib/us_data_2025/{calendars,features,instruments}"
    exit 1
fi
if [ ! -f "${US_HDF5_DIR}/daily_pv.h5" ]; then
    echo "Error: US HDF5 not found at ${US_HDF5_DIR}/daily_pv.h5"
    echo "Run: scripts/build_us_hdf5_from_parquet.py"
    exit 1
fi

# Override env vars for US mining
export QLIB_DATA_DIR="${US_QLIB_DATA}"
export QLIB_PROVIDER_URI="${US_QLIB_DATA}"
export FACTOR_CoSTEER_DATA_FOLDER="${US_HDF5_DIR}"
export FACTOR_CoSTEER_DATA_FOLDER_DEBUG="${US_HDF5_DEBUG_DIR}"
export FACTOR_CoSTEER_MARKET_REGION="us"

# Create ~/.qlib/qlib_data/us_data symlink so factor_template/us/*.yaml resolves
QLIB_SYMLINK_DIR="$HOME/.qlib/qlib_data"
mkdir -p "${QLIB_SYMLINK_DIR}"
if [ ! -L "${QLIB_SYMLINK_DIR}/us_data" ] || \
   [ "$(readlink -f ${QLIB_SYMLINK_DIR}/us_data 2>/dev/null)" != "$(readlink -f ${US_QLIB_DATA})" ]; then
    ln -sfn "${US_QLIB_DATA}" "${QLIB_SYMLINK_DIR}/us_data"
    echo "Created symlink: ${QLIB_SYMLINK_DIR}/us_data -> ${US_QLIB_DATA}"
fi

echo "US Qlib data: ${US_QLIB_DATA}"
echo "US HDF5 data: ${US_HDF5_DIR}"

# =============================================================================
# Activate conda environment
# =============================================================================
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null || \
    source activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null

if ! command -v quantaalpha &> /dev/null; then
    echo "Error: quantaalpha command not found. Please install: pip install -e ."
    exit 1
fi

# =============================================================================
# Experiment isolation
# =============================================================================
CONFIG_PATH=${CONFIG_PATH:-"configs/experiment.yaml"}

if [ -z "${EXPERIMENT_ID}" ]; then
    EXPERIMENT_ID="us_$(date +%Y%m%d_%H%M%S)"
fi
export EXPERIMENT_ID

RESULTS_BASE="${DATA_RESULTS_DIR:-./data/results}"
export WORKSPACE_PATH="${RESULTS_BASE}/workspace_${EXPERIMENT_ID}"
export PICKLE_CACHE_FOLDER_PATH_STR="${RESULTS_BASE}/pickle_cache_${EXPERIMENT_ID}"
mkdir -p "${WORKSPACE_PATH}" "${PICKLE_CACHE_FOLDER_PATH_STR}"

echo "Experiment ID: ${EXPERIMENT_ID}"
echo "Workspace: ${WORKSPACE_PATH}"
echo ""

# =============================================================================
# Parse arguments and run
# =============================================================================
DIRECTION="${1:-S&P500 Price-Volume Factor Mining}"
LIBRARY_SUFFIX="${2:-exp_us}"

export FACTOR_LIBRARY_SUFFIX="${LIBRARY_SUFFIX}"

echo "Direction: ${DIRECTION}"
echo "Library suffix: ${LIBRARY_SUFFIX}"
echo "Market region: us (S&P500)"
echo "----------------------------------------"

if [ -n "${STEP_N}" ]; then
    quantaalpha mine --direction "${DIRECTION}" --step_n "${STEP_N}" \
        --config_path "${CONFIG_PATH}" --factor_lib_suffix "${LIBRARY_SUFFIX}"
else
    quantaalpha mine --direction "${DIRECTION}" \
        --config_path "${CONFIG_PATH}" --factor_lib_suffix "${LIBRARY_SUFFIX}"
fi
