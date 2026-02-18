"""pytest conftest: inject conda env config vars so tests have Ollama API keys available.

When pytest is invoked via the env's Python binary directly (without ``conda activate``),
conda env config vars are not exported automatically.  This conftest reads them from
``conda env config vars list --json`` and sets them as os.environ fallbacks (override=False).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess


def _load_conda_env_vars(env_name: str = "quantaalpha-ollama") -> None:
    conda_exe = shutil.which("conda") or os.path.expanduser("~/anaconda3/bin/conda")
    if not os.path.exists(conda_exe):
        return
    try:
        result = subprocess.run(
            [conda_exe, "env", "config", "vars", "list", "-n", env_name, "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return
        vars_: dict[str, str] = json.loads(result.stdout)
        for key, val in vars_.items():
            os.environ.setdefault(key, val)
    except Exception:
        pass


_load_conda_env_vars()
