#!/usr/bin/env python3
"""Validate whether a DoltHub stocks repo can support paper-grade SP500 backtests."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    passed: bool
    detail: str


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout


def _parse_csv_rows(text: str) -> list[dict[str, str]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def _dolt_sql(repo_path: Path, query: str) -> list[dict[str, str]]:
    out = _run(["dolt", "sql", "-r", "csv", "-q", query], cwd=repo_path)
    return _parse_csv_rows(out)


def _clone_or_update(repo_spec: str, workdir: Path) -> Path:
    owner, repo = repo_spec.split("/", 1)
    repo_path = workdir / owner / repo
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    if not repo_path.exists():
        _run(["dolt", "clone", repo_spec], cwd=repo_path.parent)
    else:
        _run(["dolt", "pull"], cwd=repo_path)
    return repo_path


def _get_tables(repo_path: Path) -> list[str]:
    rows = _dolt_sql(repo_path, "show tables")
    if not rows:
        return []
    key = next(iter(rows[0].keys()))
    return sorted({r[key] for r in rows if r.get(key)})


def _describe_table(repo_path: Path, table: str) -> list[dict[str, str]]:
    return _dolt_sql(repo_path, f"describe `{table}`")


def _date_coverage(repo_path: Path, table: str, columns: list[str]) -> dict[str, str]:
    date_cols = [c for c in columns if "date" in c.lower() or "time" in c.lower()]
    for col in date_cols:
        try:
            rows = _dolt_sql(
                repo_path,
                f"select min(`{col}`) as min_date, max(`{col}`) as max_date from `{table}`",
            )
            if rows and rows[0].get("max_date"):
                return {"column": col, "min_date": rows[0].get("min_date", ""), "max_date": rows[0].get("max_date", "")}
        except Exception:
            continue
    return {}


def _pick_tables(tables: list[str]) -> dict[str, list[str]]:
    t_low = {t: t.lower() for t in tables}
    membership = [t for t, l in t_low.items() if "sp500" in l and any(k in l for k in ("member", "constituent", "index"))]
    if not membership:
        membership = [t for t, l in t_low.items() if "sp500" in l]
    prices = [t for t, l in t_low.items() if any(k in l for k in ("price", "ohlc", "eod", "daily"))]
    actions = [t for t, l in t_low.items() if any(k in l for k in ("split", "dividend", "action"))]
    delist = [t for t, l in t_low.items() if "delist" in l]
    return {"membership": membership, "prices": prices, "actions": actions, "delist": delist}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DoltHub SP500 reproducibility readiness")
    parser.add_argument("--repo", default="post-no-preference/stocks", help="DoltHub repo in owner/name form")
    parser.add_argument("--workdir", default="data/dolthub", help="Directory where repo will be cloned")
    parser.add_argument("--target-end", default="2025-12-26", help="Required max date for paper reproduction")
    parser.add_argument("--output", default="data/dolthub/sp500_readiness_report.json", help="Report output JSON path")
    args = parser.parse_args()

    if shutil.which("dolt") is None:
        raise SystemExit("dolt CLI not found. Install from https://github.com/dolthub/dolt and re-run.")

    workdir = Path(args.workdir)
    repo_path = _clone_or_update(args.repo, workdir)

    tables = _get_tables(repo_path)
    picked = _pick_tables(tables)
    table_profiles: dict[str, Any] = {}

    for category, cat_tables in picked.items():
        for table in cat_tables[:6]:  # cap for speed
            desc_rows = _describe_table(repo_path, table)
            cols = [r.get("Field", "") for r in desc_rows if r.get("Field")]
            cov = _date_coverage(repo_path, table, cols)
            table_profiles.setdefault(category, []).append(
                {"table": table, "columns": cols, "date_coverage": cov}
            )

    checks = {
        "has_membership_table": CheckResult(bool(picked["membership"]), ", ".join(picked["membership"][:5]) or "not found"),
        "has_price_table": CheckResult(bool(picked["prices"]), ", ".join(picked["prices"][:5]) or "not found"),
        "has_corporate_actions_table": CheckResult(bool(picked["actions"]), ", ".join(picked["actions"][:5]) or "not found"),
        "has_delisting_table": CheckResult(bool(picked["delist"]), ", ".join(picked["delist"][:5]) or "not found"),
    }

    max_dates = []
    for cat in ("prices", "membership"):
        for prof in table_profiles.get(cat, []):
            mx = prof.get("date_coverage", {}).get("max_date")
            if mx:
                max_dates.append(mx)

    has_target_coverage = any(mx >= args.target_end for mx in max_dates)
    checks["has_target_window_to_2025_12_26"] = CheckResult(
        has_target_coverage,
        f"max_dates={max_dates[:8]}",
    )

    report = {
        "repo": args.repo,
        "repo_path": str(repo_path.resolve()),
        "target_end": args.target_end,
        "tables_count": len(tables),
        "picked_tables": picked,
        "checks": {k: {"passed": v.passed, "detail": v.detail} for k, v in checks.items()},
        "profiles": table_profiles,
        "ready_for_paper_sp500": all(v.passed for v in checks.values()),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ready_for_paper_sp500": report["ready_for_paper_sp500"], "report": str(out.resolve())}))


if __name__ == "__main__":
    main()

