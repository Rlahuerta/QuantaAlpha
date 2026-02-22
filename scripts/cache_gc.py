#!/usr/bin/env python3
"""Garbage collector for QuantaAlpha cache files.

Identifies and removes:
1. Orphaned factor cache PKL files not referenced by any factor library
2. Stale workspace directories from terminated mining runs
3. Stale pickle_cache directories from terminated mining runs
4. Old MLflow experiment artifacts

Usage:
    # Dry run (default) — show what would be deleted
    python scripts/cache_gc.py

    # Actually delete
    python scripts/cache_gc.py --delete

    # Only clean factor cache
    python scripts/cache_gc.py --delete --only factor_cache

    # Only clean stale workspaces
    python scripts/cache_gc.py --delete --only workspaces
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _is_mining_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def collect_referenced_md5s(factorlib_dir: str = "data/factorlib") -> set:
    """Scan all factor library JSON files and collect referenced MD5 hashes."""
    referenced = set()
    lib_files = glob.glob(os.path.join(factorlib_dir, "all_factors_library*.json"))

    for lib_path in lib_files:
        try:
            with open(lib_path) as f:
                d = json.load(f)
            for name, v in d.get("factors", {}).items():
                expr = v.get("factor_expression", "")
                if expr:
                    md5 = hashlib.md5(expr.encode()).hexdigest()
                    referenced.add(md5)
        except Exception:
            pass

    return referenced


def gc_factor_cache(
    cache_dir: str = "data/results/factor_cache",
    factorlib_dir: str = "data/factorlib",
    delete: bool = False,
) -> tuple:
    """Remove orphaned PKL files from factor cache.

    Returns (orphan_count, orphan_bytes, kept_count, kept_bytes).
    """
    if not os.path.isdir(cache_dir):
        print(f"  Cache dir not found: {cache_dir}")
        return (0, 0, 0, 0)

    referenced = collect_referenced_md5s(factorlib_dir)
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]

    orphan_count = 0
    orphan_bytes = 0
    kept_count = 0
    kept_bytes = 0

    for fname in cache_files:
        md5 = fname.replace(".pkl", "")
        fpath = os.path.join(cache_dir, fname)
        try:
            fsize = os.path.getsize(fpath)
        except OSError:
            continue

        if md5 not in referenced:
            orphan_count += 1
            orphan_bytes += fsize
            if delete:
                os.remove(fpath)
        else:
            kept_count += 1
            kept_bytes += fsize

    return (orphan_count, orphan_bytes, kept_count, kept_bytes)


def gc_factor_cache_us(
    cache_dir: str = "data/results/factor_cache_us",
    factorlib_dir: str = "data/factorlib",
    delete: bool = False,
) -> tuple:
    """Same as gc_factor_cache but for US cache."""
    return gc_factor_cache(cache_dir, factorlib_dir, delete)


def _extract_pid_from_dir(dirname: str) -> int | None:
    """Try to extract a PID or timestamp from workspace/pickle_cache dir names."""
    # These dirs don't embed PIDs, but we can check by age
    return None


def gc_stale_workspaces(
    results_dir: str = "data/results",
    max_age_hours: int = 48,
    delete: bool = False,
) -> tuple:
    """Remove stale workspace_* and pickle_cache_* directories.

    A directory is considered stale if:
    - It's older than max_age_hours AND
    - No mining process references it (checked via /proc)

    Returns (removed_count, removed_bytes).
    """
    if not os.path.isdir(results_dir):
        return (0, 0)

    removed_count = 0
    removed_bytes = 0
    cutoff = datetime.now() - timedelta(hours=max_age_hours)

    # Also check which workspaces are actively in use by mining processes
    active_workspace_patterns = set()
    try:
        import subprocess

        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "quantaalpha mine" in line or "run_backtest" in line:
                # Extract any workspace references
                for part in line.split():
                    if "workspace" in part or "pickle_cache" in part:
                        active_workspace_patterns.add(os.path.basename(part))
    except Exception:
        pass

    for entry in os.listdir(results_dir):
        full_path = os.path.join(results_dir, entry)
        if not os.path.isdir(full_path):
            continue

        # Only target workspace_* and pickle_cache_* dirs
        if not (entry.startswith("workspace_") or entry.startswith("pickle_cache_")):
            continue

        # Skip if actively referenced
        if entry in active_workspace_patterns:
            continue

        # Check modification time
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(full_path))
        except OSError:
            continue

        if mtime < cutoff:
            dir_size = _dir_size(full_path)
            removed_count += 1
            removed_bytes += dir_size
            if delete:
                shutil.rmtree(full_path, ignore_errors=True)

    return (removed_count, removed_bytes)


def gc_mlruns(
    mlruns_dir: str = "mlruns",
    max_age_hours: int = 168,  # 7 days
    delete: bool = False,
) -> tuple:
    """Remove old MLflow experiment directories.

    Returns (removed_count, removed_bytes).
    """
    if not os.path.isdir(mlruns_dir):
        return (0, 0)

    removed_count = 0
    removed_bytes = 0
    cutoff = datetime.now() - timedelta(hours=max_age_hours)

    for exp_id in os.listdir(mlruns_dir):
        exp_path = os.path.join(mlruns_dir, exp_id)
        if not os.path.isdir(exp_path):
            continue
        if exp_id in (".trash", "models"):
            continue

        # Check if experiment has recent runs
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(exp_path))
        except OSError:
            continue

        if mtime < cutoff:
            dir_size = _dir_size(exp_path)
            removed_count += 1
            removed_bytes += dir_size
            if delete:
                shutil.rmtree(exp_path, ignore_errors=True)

    return (removed_count, removed_bytes)


def main():
    parser = argparse.ArgumentParser(description="QuantaAlpha cache garbage collector")
    parser.add_argument(
        "--delete", action="store_true", help="Actually delete files (default: dry run)"
    )
    parser.add_argument(
        "--only",
        choices=["factor_cache", "workspaces", "mlruns", "all"],
        default="all",
        help="Only clean specific cache type",
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=48,
        help="Max age in hours for stale workspaces (default: 48)",
    )
    parser.add_argument(
        "--factorlib-dir",
        default="data/factorlib",
        help="Factor library directory",
    )
    args = parser.parse_args()

    mode = "DELETE" if args.delete else "DRY RUN"
    print(f"{'='*60}")
    print(f"QuantaAlpha Cache Garbage Collector [{mode}]")
    print(f"{'='*60}")
    print()

    total_reclaimable = 0

    # 1. Factor cache (CN)
    if args.only in ("factor_cache", "all"):
        print("📦 Factor cache (CN): data/results/factor_cache/")
        orphans, orphan_bytes, kept, kept_bytes = gc_factor_cache(
            "data/results/factor_cache", args.factorlib_dir, args.delete
        )
        total_reclaimable += orphan_bytes
        print(f"   Kept: {kept} files ({_human_size(kept_bytes)})")
        print(f"   Orphaned: {orphans} files ({_human_size(orphan_bytes)})")
        if args.delete and orphans > 0:
            print(f"   ✅ Deleted {orphans} orphaned files")
        print()

        # Factor cache (US)
        print("📦 Factor cache (US): data/results/factor_cache_us/")
        orphans_us, orphan_bytes_us, kept_us, kept_bytes_us = gc_factor_cache_us(
            "data/results/factor_cache_us", args.factorlib_dir, args.delete
        )
        total_reclaimable += orphan_bytes_us
        print(f"   Kept: {kept_us} files ({_human_size(kept_bytes_us)})")
        print(f"   Orphaned: {orphans_us} files ({_human_size(orphan_bytes_us)})")
        if args.delete and orphans_us > 0:
            print(f"   ✅ Deleted {orphans_us} orphaned files")
        print()

    # 2. Stale workspaces
    if args.only in ("workspaces", "all"):
        print(f"🗂️  Stale workspaces (>{args.max_age_hours}h): data/results/")
        removed, removed_bytes = gc_stale_workspaces(
            "data/results", args.max_age_hours, args.delete
        )
        total_reclaimable += removed_bytes
        print(f"   Stale dirs: {removed} ({_human_size(removed_bytes)})")
        if args.delete and removed > 0:
            print(f"   ✅ Removed {removed} stale directories")
        print()

    # 3. MLflow runs
    if args.only in ("mlruns", "all"):
        print("📊 MLflow experiments (>7 days): mlruns/")
        removed_ml, removed_ml_bytes = gc_mlruns("mlruns", 168, args.delete)
        total_reclaimable += removed_ml_bytes
        print(f"   Stale experiments: {removed_ml} ({_human_size(removed_ml_bytes)})")
        if args.delete and removed_ml > 0:
            print(f"   ✅ Removed {removed_ml} stale experiments")
        print()

    # Summary
    print(f"{'='*60}")
    action = "Reclaimed" if args.delete else "Reclaimable"
    print(f"💾 {action}: {_human_size(total_reclaimable)}")
    if not args.delete and total_reclaimable > 0:
        print(f"   Run with --delete to reclaim space")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
