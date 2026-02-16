#!/usr/bin/env python3
"""
Per-module coverage gate for frequently touched risk-heavy modules.

Usage (after `coverage run -m pytest`):
  python tests/coverage_gate.py
"""

from __future__ import annotations

import subprocess
import sys


MODULE_THRESHOLDS = {
    "quantaalpha/backtest/runner.py": 40,
    "quantaalpha/backtest/factor_loader.py": 50,
    "quantaalpha/factors/coder/expr_parser.py": 60,
    "quantaalpha/llm/client.py": 25,
    "quantaalpha/factors/coder/function_lib.py": 26,
    "quantaalpha/factors/regulator/factor_regulator.py": 20,
}


def main() -> int:
    failed = False
    for module_path, threshold in MODULE_THRESHOLDS.items():
        cmd = [
            sys.executable,
            "-m",
            "coverage",
            "report",
            "--include",
            module_path,
            "--fail-under",
            str(threshold),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.stdout:
            print(result.stdout.strip())
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr.strip(), file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
