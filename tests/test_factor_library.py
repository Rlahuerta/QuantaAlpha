"""
Tests for quantaalpha/factors/library.py — FactorLibraryManager.

Covers:
- Bug B-save: add_factors_from_experiment() must call _save() (file persisted to disk)
- Bug B-quality: quality classification must be set on every factor entry
- _save() metadata counts (high/medium/low_quality_count)
- Corrupt / missing file graceful fallback
- Deduplication by factor_id (same name+expr overwrites)
- None experiment guard
- check_cache_status returns correct totals
- warm_cache_from_json skips factors with no source
"""

import hashlib
import json
import types
from pathlib import Path

import pytest

from quantaalpha.factors.library import FactorLibraryManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(name: str, expr: str, desc: str = "") -> types.SimpleNamespace:
    t = types.SimpleNamespace()
    t.factor_name = name
    t.factor_expression = expr
    t.factor_description = desc
    t.factor_formulation = ""
    return t


def _make_experiment(tasks, rank_ic: float = 0.025, ic: float = 0.01) -> types.SimpleNamespace:
    """Minimal fake QlibFactorExperiment."""
    exp = types.SimpleNamespace()
    exp.sub_tasks = tasks
    exp.sub_workspace_list = []
    # _extract_backtest_results reads experiment.result (dict/Series/DataFrame)
    exp.result = {"Rank IC": rank_ic, "IC": ic}
    return exp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def lib_path(tmp_path) -> Path:
    return tmp_path / "test_library.json"


@pytest.fixture()
def mgr(lib_path) -> FactorLibraryManager:
    return FactorLibraryManager(str(lib_path))


# ---------------------------------------------------------------------------
# Bug B-save: file must exist on disk after add_factors_from_experiment
# ---------------------------------------------------------------------------

def test_add_factors_writes_file_to_disk(mgr, lib_path):
    """Regression: _save() was never called — file stayed in memory only."""
    exp = _make_experiment([_make_task("MomentumFactor", "TS_MEAN($return,20)")])
    mgr.add_factors_from_experiment(exp)
    assert lib_path.exists(), "Library file must be written to disk after add_factors_from_experiment()"


def test_add_factors_file_is_valid_json(mgr, lib_path):
    exp = _make_experiment([_make_task("MomentumFactor", "TS_MEAN($return,20)")])
    mgr.add_factors_from_experiment(exp)
    data = json.loads(lib_path.read_text())
    assert "metadata" in data
    assert "factors" in data
    assert len(data["factors"]) == 1


# ---------------------------------------------------------------------------
# Bug B-quality: quality field must be set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rank_ic, ic, expected_quality", [
    (0.025, 0.005, "high_quality"),   # rank_ic >= 0.02
    (0.005, 0.025, "high_quality"),   # ic >= 0.02
    (0.015, 0.005, "medium_quality"), # 0.01 <= rank_ic < 0.02
    (0.005, 0.015, "medium_quality"), # 0.01 <= ic < 0.02
    (0.005, 0.005, "low_quality"),    # both below 0.01
    (0.0,   0.0,   "low_quality"),    # zeros
])
def test_quality_classification(mgr, lib_path, rank_ic, ic, expected_quality):
    """Regression: quality was None — classification logic was missing."""
    exp = _make_experiment([_make_task("F", "TS_MEAN($close,5)")], rank_ic=rank_ic, ic=ic)
    mgr.add_factors_from_experiment(exp)
    data = json.loads(lib_path.read_text())
    factor = next(iter(data["factors"].values()))
    assert factor["quality"] == expected_quality, (
        f"rank_ic={rank_ic}, ic={ic} → expected {expected_quality}, got {factor['quality']}"
    )


# ---------------------------------------------------------------------------
# _save() metadata quality counts
# ---------------------------------------------------------------------------

def test_save_updates_quality_counts(mgr, lib_path):
    """_save() must write high/medium/low_quality_count into metadata."""
    tasks = [
        _make_task("HighF1", "TS_MEAN($return,20)"),
        _make_task("HighF2", "TS_STD($return,20)"),
    ]
    exp = _make_experiment(tasks, rank_ic=0.03, ic=0.03)
    mgr.add_factors_from_experiment(exp)

    # Add a low-quality one separately
    exp2 = _make_experiment([_make_task("LowF", "ABS($return)")], rank_ic=0.001, ic=0.001)
    mgr.add_factors_from_experiment(exp2)

    data = json.loads(lib_path.read_text())
    meta = data["metadata"]
    assert meta["total_factors"] == 3
    assert meta["high_quality_count"] == 2
    assert meta["low_quality_count"] == 1
    assert meta["medium_quality_count"] == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_same_factor_overwrites_not_duplicates(mgr, lib_path):
    """Same factor_name + expression produces same MD5 → only one entry."""
    task = _make_task("VolumeMomentum", "TS_MEAN($volume,10)")
    exp = _make_experiment([task])
    mgr.add_factors_from_experiment(exp)
    mgr.add_factors_from_experiment(exp)  # second add, same factor

    data = json.loads(lib_path.read_text())
    assert len(data["factors"]) == 1
    assert data["metadata"]["total_factors"] == 1


# ---------------------------------------------------------------------------
# None experiment guard
# ---------------------------------------------------------------------------

def test_none_experiment_does_not_write(mgr, lib_path):
    mgr.add_factors_from_experiment(None)
    assert not lib_path.exists(), "No file should be written for None experiment"


# ---------------------------------------------------------------------------
# Persistence across reload
# ---------------------------------------------------------------------------

def test_reload_reads_existing_library(lib_path):
    """New FactorLibraryManager instance loads previously saved factors."""
    mgr1 = FactorLibraryManager(str(lib_path))
    exp = _make_experiment([_make_task("Factor1", "RANK($close)")])
    mgr1.add_factors_from_experiment(exp)

    mgr2 = FactorLibraryManager(str(lib_path))
    assert len(mgr2.data["factors"]) == 1


# ---------------------------------------------------------------------------
# Corrupt file fallback
# ---------------------------------------------------------------------------

def test_corrupt_file_falls_back_to_empty(lib_path):
    lib_path.write_text("{{not valid json}}")
    mgr = FactorLibraryManager(str(lib_path))
    assert mgr.data["factors"] == {}
    assert mgr.data["metadata"]["total_factors"] == 0


# ---------------------------------------------------------------------------
# Multiple tasks in one experiment
# ---------------------------------------------------------------------------

def test_multiple_tasks_all_saved(mgr, lib_path):
    tasks = [_make_task(f"Factor{i}", f"TS_MEAN($return,{i+5})") for i in range(5)]
    exp = _make_experiment(tasks, rank_ic=0.025)
    mgr.add_factors_from_experiment(exp)

    data = json.loads(lib_path.read_text())
    assert len(data["factors"]) == 5
    assert data["metadata"]["total_factors"] == 5


# ---------------------------------------------------------------------------
# check_cache_status
# ---------------------------------------------------------------------------

def test_check_cache_status_no_h5(lib_path):
    mgr = FactorLibraryManager(str(lib_path))
    exp = _make_experiment([_make_task("F1", "TS_MEAN($close,5)")])
    mgr.add_factors_from_experiment(exp)

    result = FactorLibraryManager.check_cache_status(str(lib_path))
    assert result["total"] == 1
    assert result["need_compute"] == 1
    assert result["h5_cached"] == 0


# ---------------------------------------------------------------------------
# warm_cache_from_json — no-source factors are skipped
# ---------------------------------------------------------------------------

def test_warm_cache_skips_no_source(lib_path, tmp_path):
    mgr = FactorLibraryManager(str(lib_path))
    # Factor with expression but no h5 path
    exp = _make_experiment([_make_task("F", "RANK($volume)")])
    mgr.add_factors_from_experiment(exp)

    result = FactorLibraryManager.warm_cache_from_json(str(lib_path), cache_dir=str(tmp_path / "cache"))
    assert result["no_source"] == 1
    assert result["skipped"] == 1
    assert result["synced"] == 0


# ---------------------------------------------------------------------------
# factor_id is stable (deterministic MD5)
# ---------------------------------------------------------------------------

def test_factor_id_is_deterministic(mgr, lib_path):
    name, expr = "TestFactor", "TS_MEAN($return,10)"
    expected_id = hashlib.md5(f"{name}_{expr}".encode()).hexdigest()[:16]

    exp = _make_experiment([_make_task(name, expr)])
    mgr.add_factors_from_experiment(exp)

    data = json.loads(lib_path.read_text())
    assert expected_id in data["factors"]
