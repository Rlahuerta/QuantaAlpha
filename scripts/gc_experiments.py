#!/usr/bin/env python3
"""
Garbage collector for QuantaAlpha experiment artifacts.

Cleans up old workspace/, pickle_cache/, factor_cache/, and log/ worker caches
while preserving artifacts from currently running experiments.

Usage:
    # Dry-run (safe default — only prints what would be deleted):
    python scripts/gc_experiments.py

    # Delete everything eligible (keeps last N experiments):
    python scripts/gc_experiments.py --execute --keep-n 2

    # Aggressive: also clean log worker caches and factor caches
    python scripts/gc_experiments.py --execute --keep-n 1 --all

    # Clean only log worker caches from finished runs:
    python scripts/gc_experiments.py --execute --only log-workers
"""
import argparse
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results"
LOG_DIR = REPO_ROOT / "log"
PICKLE_ROOT = REPO_ROOT / "pickle_cache"          # repo-root tiny metadata cache

# Dirs inside data/results/ that are considered "legacy" (no timestamp suffix)
LEGACY_SUBDIRS = ["workspace", "factor_cache", "factor_cache_us"]
# Sub-namespaces inside data/results/pickle_cache/ that grow unboundedly
LEGACY_PICKLE_NAMESPACES = [
    "quantaalpha.factors.coder.factor.execute",
    "quantaalpha.factors.runner.develop",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")   # e.g. 20260219_191415


def parse_timestamp(name: str) -> datetime | None:
    m = TIMESTAMP_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fmt_size(path: Path) -> str:
    """Return human-readable size of a path."""
    try:
        result = subprocess.run(
            ["du", "-sh", "--apparent-size", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.split()[0] if result.returncode == 0 else "?"
    except Exception:
        return "?"


def get_active_experiment_ids() -> set[str]:
    """
    Read running `quantaalpha mine` processes and extract their experiment
    workspace timestamps so we never delete an active experiment.
    """
    active: set[str] = set()
    try:
        out = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in out.splitlines():
            if "quantaalpha mine" not in line:
                continue
            for m in TIMESTAMP_RE.finditer(line):
                active.add(m.group(1))
    except Exception:
        pass
    # Also collect IDs from dirs still open by running processes
    return active


def _dirs_by_timestamp(pattern: str) -> list[tuple[datetime, Path]]:
    """Return (timestamp, path) pairs sorted newest-first for glob pattern."""
    results = []
    for p in RESULTS_DIR.glob(pattern):
        if not p.is_dir():
            continue
        ts = parse_timestamp(p.name)
        if ts:
            results.append((ts, p))
    return sorted(results, key=lambda x: x[0], reverse=True)


def _delete(path: Path, dry_run: bool) -> None:
    size = fmt_size(path)
    if dry_run:
        print(f"  [DRY-RUN] would delete {path}  ({size})")
    else:
        print(f"  Deleting {path}  ({size}) ...", flush=True)
        shutil.rmtree(path, ignore_errors=True)
        print(f"  ✓ deleted")


# ---------------------------------------------------------------------------
# GC routines
# ---------------------------------------------------------------------------

def gc_legacy_dirs(dry_run: bool) -> None:
    """Remove non-timestamped legacy workspace/factor_cache dirs."""
    found_any = False
    for name in LEGACY_SUBDIRS:
        p = RESULTS_DIR / name
        if p.exists():
            found_any = True
            print(f"\n[legacy] {p.relative_to(REPO_ROOT)}")
            _delete(p, dry_run)
    for ns in LEGACY_PICKLE_NAMESPACES:
        p = RESULTS_DIR / "pickle_cache" / ns
        if p.exists():
            found_any = True
            print(f"\n[legacy pickle] {p.relative_to(REPO_ROOT)}")
            _delete(p, dry_run)
    if not found_any:
        print("  No legacy dirs found.")


def gc_timestamped_experiments(keep_n: int, dry_run: bool, active_ids: set[str]) -> None:
    """
    Keep the `keep_n` most-recent timestamped workspace/pickle_cache dirs;
    delete the rest.  Never deletes dirs whose timestamp appears in active_ids.
    """
    # Collect unique timestamps across all experiment dirs
    all_ts: dict[str, list[Path]] = {}  # timestamp → [workspace, pickle_cache, ...]
    for p in RESULTS_DIR.iterdir():
        if not p.is_dir():
            continue
        m = TIMESTAMP_RE.search(p.name)
        if not m:
            continue
        ts = m.group(1)
        all_ts.setdefault(ts, []).append(p)

    sorted_ts = sorted(all_ts.keys(), reverse=True)   # newest first
    print(f"\n[timestamped experiments] found {len(sorted_ts)} timestamps; keeping {keep_n}")

    for i, ts in enumerate(sorted_ts):
        dirs = all_ts[ts]
        if ts in active_ids:
            print(f"  SKIP (active)  {ts}: {[d.name for d in dirs]}")
            continue
        if i < keep_n:
            print(f"  KEEP (recent)  {ts}: {[d.name for d in dirs]}")
            continue
        print(f"  → {ts}:")
        for d in dirs:
            _delete(d, dry_run)


def gc_log_worker_caches(dry_run: bool, active_ids: set[str]) -> None:
    """
    Delete pickle_cache_N/ subdirs from log experiment dirs whose timestamp
    is NOT in active_ids (i.e. the run has finished).
    """
    LOG_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
    found_any = False

    for log_exp in sorted(LOG_DIR.iterdir()):
        if not log_exp.is_dir():
            continue
        m = LOG_TS_RE.match(log_exp.name)
        if not m:
            continue

        # Convert log timestamp (YYYY-MM-DD_HH-MM-SS) to results timestamp (YYYYMMDD_HHMMSS)
        raw = m.group(1).replace("-", "").replace("_", "_")
        date_part, time_part = raw.split("_")
        results_ts = date_part + "_" + time_part.replace("-", "")

        worker_caches = sorted(log_exp.glob("pickle_cache_*"))
        if not worker_caches:
            continue

        if results_ts in active_ids:
            print(f"  SKIP (active run) {log_exp.name}: {len(worker_caches)} worker caches")
            continue

        found_any = True
        print(f"\n[log worker caches] {log_exp.name}  (run finished)")
        for wc in worker_caches:
            _delete(wc, dry_run)

    if not found_any:
        print("  No finished-run worker caches found.")


def gc_root_pickle_cache(dry_run: bool) -> None:
    """Clean the tiny repo-root pickle_cache/ (session metadata, usually <1 MB each)."""
    dirs = [p for p in PICKLE_ROOT.iterdir() if p.is_dir()] if PICKLE_ROOT.exists() else []
    if not dirs:
        print("  Root pickle_cache/ is empty.")
        return
    total = sum(p.stat().st_size for p in PICKLE_ROOT.rglob("*") if p.is_file())
    print(f"  Root pickle_cache/ holds {len(dirs)} sub-dirs ({total // 1024} KB) — keeping.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QuantaAlpha experiment garbage collector")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete files (default: dry-run only)")
    parser.add_argument("--keep-n", type=int, default=2,
                        help="Number of most-recent timestamped experiments to keep (default: 2)")
    parser.add_argument("--all", action="store_true",
                        help="Also clean legacy dirs and log worker caches")
    parser.add_argument("--only", choices=["legacy", "timestamped", "log-workers"],
                        help="Clean only one category")
    args = parser.parse_args()

    dry_run = not args.execute
    if dry_run:
        print("=== DRY-RUN MODE — pass --execute to actually delete ===\n")
    else:
        print("=== EXECUTE MODE — deleting files ===\n")

    active_ids = get_active_experiment_ids()
    print(f"Active experiment timestamps detected: {active_ids or '(none)'}\n")

    before = shutil.disk_usage(REPO_ROOT)

    if args.only == "legacy":
        gc_legacy_dirs(dry_run)
    elif args.only == "timestamped":
        gc_timestamped_experiments(args.keep_n, dry_run, active_ids)
    elif args.only == "log-workers":
        gc_log_worker_caches(dry_run, active_ids)
    else:
        # Default: all three
        print("─── Legacy dirs ─────────────────────────────")
        gc_legacy_dirs(dry_run)
        print("\n─── Timestamped experiments ─────────────────")
        gc_timestamped_experiments(args.keep_n, dry_run, active_ids)
        print("\n─── Log worker caches ───────────────────────")
        gc_log_worker_caches(dry_run, active_ids)
        print("\n─── Root pickle_cache/ ──────────────────────")
        gc_root_pickle_cache(dry_run)

    after = shutil.disk_usage(REPO_ROOT)
    freed = (before.used - after.used) / (1024 ** 3)
    pct = 100 * after.used / after.total
    if args.execute:
        print(f"\n✓ Freed {freed:.1f} GB  |  Disk now {pct:.0f}% full ({after.free / 1024**3:.0f} GB free)")
    else:
        print(f"\nCurrent disk: {pct:.0f}% full ({after.free / 1024**3:.0f} GB free)")
        print("Run with --execute to actually free space.")


if __name__ == "__main__":
    main()
