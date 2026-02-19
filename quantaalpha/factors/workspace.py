"""
QuantaAlpha custom workspace.

Overrides rdagent QlibFBWorkspace: project-level factor_template overrides default YAML;
base files (read_exp_res.py, etc.) still from rdagent; init empty git repo in workspace to suppress qlib recorder git output.
Uses a conda-aware LocalEnv so that qrun and python both resolve to the active conda venv.
"""

import re
import subprocess
import sys
import os
from pathlib import Path

import pandas as pd
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace as _RdagentQlibFBWorkspace
from rdagent.utils.env import LocalEnv, LocalConf
from rdagent.log import rdagent_logger as logger

_CUSTOM_TEMPLATE_DIR = Path(__file__).resolve().parent / "factor_template"

_MARKET_REGION_TEMPLATE_DIRS: dict[str, Path] = {
    "cn": _CUSTOM_TEMPLATE_DIR,
    "us": _CUSTOM_TEMPLATE_DIR / "us",
}


def _make_conda_local_env() -> LocalEnv:
    """Return a LocalEnv whose bin_path points to the active conda venv bin dir.

    rdagent's QlibCondaConf detects the bin path by running
    ``conda run -n rdagent4qlib env | grep PATH``, but *conda* is not on PATH in
    this environment, so detection fails and bin_path stays "".  As a result
    LocalEnv prepends only ``/bin/:/usr/bin/`` and bare ``python`` / ``qrun``
    resolve to the system Python (which has no packages).

    Resolution order for the Python executable:
    1. ``VENV_PYTHON`` env var  — explicit absolute path (set in .env)
    2. ``CONDA_ENV_NAME`` env var — derive via ``<conda_base>/envs/<name>/bin/python``
    3. ``sys.executable`` — active interpreter (reliable when launched from the venv)
    """
    python_exe: str | None = None

    # 1. Explicit override from .env
    python_exe = os.environ.get("VENV_PYTHON", "").strip() or None

    # 2. Derive from CONDA_ENV_NAME if VENV_PYTHON is absent
    if not python_exe:
        env_name = os.environ.get("CONDA_ENV_NAME", "").strip()
        if env_name:
            # Try common conda base locations
            for base in [
                os.environ.get("CONDA_PREFIX", ""),
                str(Path.home() / "anaconda3"),
                str(Path.home() / "miniconda3"),
                "/opt/conda",
            ]:
                if not base:
                    continue
                # base might itself be the active env; walk up to find base/envs/<name>
                candidate = Path(base) / "envs" / env_name / "bin" / "python"
                if candidate.exists():
                    python_exe = str(candidate)
                    break
                # also try base directly (if CONDA_PREFIX is already the target env)
                candidate2 = Path(base).parent.parent / "envs" / env_name / "bin" / "python"
                if candidate2.exists():
                    python_exe = str(candidate2)
                    break

    # 3. Fall back to the current interpreter
    if not python_exe:
        python_exe = sys.executable

    conda_bin = str(Path(python_exe).parent)
    conf = LocalConf(bin_path=conda_bin, default_entry=f"{python_exe} main.py")
    return LocalEnv(conf=conf)


class QlibFBWorkspace(_RdagentQlibFBWorkspace):
    """
    Override rdagent QlibFBWorkspace: inject project factor_template/ YAML over defaults;
    init empty git repo in workspace to avoid qlib recorder git help output.
    """

    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(template_folder_path, *args, **kwargs)
        if _CUSTOM_TEMPLATE_DIR.exists():
            self.inject_code_from_folder(_CUSTOM_TEMPLATE_DIR)
            logger.info(f"Overrode rdagent default config with project template: {_CUSTOM_TEMPLATE_DIR}")
        # If a non-default market region is configured, inject market-specific overrides
        # after the base templates so they win over the CN defaults.
        from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
        region = FACTOR_COSTEER_SETTINGS.market_region.lower()
        if region != "cn":
            region_dir = _MARKET_REGION_TEMPLATE_DIRS.get(region)
            if region_dir and region_dir.exists():
                self.inject_code_from_folder(region_dir)
                logger.info(f"Injected {region} market templates from {region_dir}")

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {}, *args, **kwargs):
        """Execute qlib backtest using a conda-aware LocalEnv.

        Bypasses rdagent's QlibCondaEnv (which tries to resolve conda env PATH
        via ``conda run -n rdagent4qlib`` and silently falls back to an empty
        bin_path when conda is not on PATH).  Instead, creates a LocalEnv with
        bin_path = the venv bin directory so that ``python`` and ``qrun`` both
        resolve to the correct conda venv binaries.

        The Python executable is resolved from (in order):
          VENV_PYTHON env var → CONDA_ENV_NAME env var → sys.executable
        """
        qtde = _make_conda_local_env()
        # Disable LocalEnv.cached_run: its cache key excludes .parquet files, so all directions
        # with identical .py files share the same cache key.  On a cache hit the cached workspace
        # zip (from direction 0) is unzipped over the current workspace, overwriting
        # combined_factors_df.parquet with stale factor data from a different direction.
        qtde.conf.enable_cache = False
        qtde.prepare()

        # Use the same python that _make_conda_local_env resolved
        python_exe = qtde.conf.default_entry.split()[0]  # "<python_exe> main.py"

        run_env = dict(run_env)  # copy; do not mutate caller's dict

        # Run qlib backtest
        execute_qlib_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=run_env,
        )
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")

        # Read experiment results using the resolved venv Python
        execute_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry=f"{python_exe} read_exp_res.py",
            env=run_env,
        )

        quantitative_backtesting_chart_path = self.workspace_path / "ret.pkl"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
        else:
            logger.error("No result file found.")
            return None, execute_qlib_log

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            pattern = r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)"
            matches = re.findall(pattern, execute_qlib_log)
            execute_qlib_log = "\n".join(matches)
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"File {qlib_res_path} does not exist.")
            return None, execute_qlib_log

    def before_execute(self) -> None:
        """Init workspace prerequisites before qlib backtest execution."""
        super().before_execute()

        # Ensure qlib default provider path exists when mining is launched without run.sh.
        qlib_data_dir = os.environ.get("QLIB_DATA_DIR") or os.environ.get("QLIB_PROVIDER_URI")
        if qlib_data_dir:
            source = Path(qlib_data_dir).expanduser().resolve()
            target = Path.home() / ".qlib" / "qlib_data" / "cn_data"
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    if target.resolve() != source:
                        target.unlink()
                        target.symlink_to(source)
                elif not target.exists():
                    target.symlink_to(source)
                elif target.resolve() != source:
                    logger.warning(f"qlib symlink target differs and is not a symlink: {target}")
            else:
                logger.warning(f"QLIB data path does not exist: {source}")

        git_dir = self.workspace_path / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=str(self.workspace_path),
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass
