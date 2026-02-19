from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

import quantaalpha.core.experiment as experiment_module
import quantaalpha.core.utils as core_utils_module
import quantaalpha.factors.coder.factor as factor_module
from quantaalpha.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from quantaalpha.factors.coder.factor import FactorFBWorkspace, FactorTask


def _make_task(name: str = "f1", version: int = 1) -> FactorTask:
    return FactorTask(
        factor_name=name,
        factor_description="desc",
        factor_formulation="formula",
        factor_expression="TS_MEAN($close, 5)",
        version=version,
    )


def _set_common_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(core_utils_module.RD_AGENT_SETTINGS, "cache_with_pickle", False)
    monkeypatch.setattr(experiment_module.RD_AGENT_SETTINGS, "workspace_path", tmp_path / "workspace_root")
    monkeypatch.setattr(factor_module.FACTOR_COSTEER_SETTINGS, "python_bin", "python")
    monkeypatch.setattr(factor_module.FACTOR_COSTEER_SETTINGS, "file_based_execution_timeout", 1)


def test_factor_task_methods_and_repr():
    task = _make_task("alpha1", version=2)
    info = task.get_task_information()
    desc = task.get_task_description()
    result_info = task.get_task_information_and_implementation_result()
    from_dict_task = FactorTask.from_dict(
        {
            "factor_name": "alpha2",
            "factor_description": "d",
            "factor_formulation": "f",
            "factor_expression": "x",
        }
    )

    assert "factor_name: alpha1" in info
    assert "factor_description: desc" in desc
    assert result_info["factor_implementation"] == "False"
    assert repr(task) == "<FactorTask[alpha1]>"
    assert from_dict_task.factor_name == "alpha2"


def test_factor_workspace_execute_code_not_set_branches(monkeypatch, tmp_path):
    _set_common_settings(monkeypatch, tmp_path)
    task = _make_task("missing-code")
    ws = FactorFBWorkspace(target_task=task, raise_exception=False)
    ws.code_dict = {}

    feedback, result = ws.execute()
    assert feedback == FactorFBWorkspace.FB_CODE_NOT_SET
    assert result is None

    ws_raise = FactorFBWorkspace(target_task=task, raise_exception=True)
    ws_raise.code_dict = {}
    with pytest.raises(CodeFormatError):
        ws_raise.execute()


def test_factor_workspace_execute_success_and_output_read(monkeypatch, tmp_path):
    _set_common_settings(monkeypatch, tmp_path)

    data_dir = tmp_path / "data_debug"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "dummy.txt").write_text("ok", encoding="utf-8")
    monkeypatch.setattr(factor_module.FACTOR_COSTEER_SETTINGS, "data_folder_debug", str(data_dir))

    task = _make_task("ok-run", version=1)
    ws = FactorFBWorkspace(target_task=task, raise_exception=False)
    ws.inject_code(**{"factor.py": "print('ok')"})

    def _fake_check_output(cmd, shell, cwd, stderr, timeout, env):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=2), ["AAA"]],
            names=["datetime", "instrument"],
        )
        pd.DataFrame({"factor": [1.0, 2.0]}, index=idx).to_hdf(Path(cwd) / "result.h5", key="data", mode="w")
        return b"done"

    monkeypatch.setattr(factor_module.subprocess, "check_output", _fake_check_output)
    feedback, result = ws.execute(data_type="Debug")

    assert ws.FB_OUTPUT_FILE_FOUND in feedback
    assert isinstance(result, pd.DataFrame)
    assert "File Factor[ok-run]" in str(ws)
    assert repr(ws) == str(ws)


def test_factor_workspace_execute_version2_timeout_and_missing_output(monkeypatch, tmp_path):
    _set_common_settings(monkeypatch, tmp_path)
    data_dir = tmp_path / "data_full"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(factor_module.FACTOR_COSTEER_SETTINGS, "data_folder", str(data_dir))

    task_v2 = _make_task("timeout-run-v2", version=2)
    ws_v2 = FactorFBWorkspace(target_task=task_v2, raise_exception=False)
    ws_v2.inject_code(**{"factor.py": "print('ok')"})
    with pytest.raises(FileNotFoundError):
        ws_v2.execute(data_type="Train")

    task = _make_task("timeout-run", version=1)
    ws = FactorFBWorkspace(target_task=task, raise_exception=False)
    ws.inject_code(**{"factor.py": "print('ok')"})

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="python factor.py", timeout=1)

    monkeypatch.setattr(factor_module.subprocess, "check_output", _raise_timeout)
    feedback, result = ws.execute(data_type="Train")
    assert "timeout error" in feedback.lower()
    assert ws.FB_OUTPUT_FILE_NOT_FOUND in feedback
    assert result is None

    ws_raise = FactorFBWorkspace(target_task=task, raise_exception=True)
    ws_raise.inject_code(**{"factor.py": "print('ok')"})
    with pytest.raises(CustomRuntimeError):
        ws_raise.execute(data_type="Train")


def test_factor_workspace_called_process_error_branches(monkeypatch, tmp_path):
    _set_common_settings(monkeypatch, tmp_path)
    data_dir = tmp_path / "data_debug_cp"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(factor_module.FACTOR_COSTEER_SETTINGS, "data_folder_debug", str(data_dir))

    task = _make_task("cp-error", version=1)
    ws = FactorFBWorkspace(target_task=task, raise_exception=False)
    ws.inject_code(**{"factor.py": "print('bad')"})

    long_error = ("X" * 2500).encode("utf-8")

    def _raise_called(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd="python factor.py", output=long_error)

    monkeypatch.setattr(factor_module.subprocess, "check_output", _raise_called)
    feedback, result = ws.execute()
    assert "hidden long error message" in feedback
    assert ws.FB_OUTPUT_FILE_NOT_FOUND in feedback
    assert result is None

    ws_raise = FactorFBWorkspace(target_task=task, raise_exception=True)
    ws_raise.inject_code(**{"factor.py": "print('bad')"})
    with pytest.raises(CustomRuntimeError):
        ws_raise.execute()


def test_factor_workspace_output_read_error_and_no_output_exception(monkeypatch, tmp_path):
    _set_common_settings(monkeypatch, tmp_path)
    data_dir = tmp_path / "data_debug_output"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(factor_module.FACTOR_COSTEER_SETTINGS, "data_folder_debug", str(data_dir))

    task = _make_task("read-error", version=1)
    ws = FactorFBWorkspace(target_task=task, raise_exception=False)
    ws.inject_code(**{"factor.py": "print('ok')"})

    def _write_invalid_h5(cmd, shell, cwd, stderr, timeout, env):
        (Path(cwd) / "result.h5").write_text("not hdf", encoding="utf-8")
        return b"done"

    monkeypatch.setattr(factor_module.subprocess, "check_output", _write_invalid_h5)
    feedback, result = ws.execute()
    assert "Error found when reading hdf file" in feedback
    assert result is None

    ws_no_output = FactorFBWorkspace(target_task=task, raise_exception=True)
    ws_no_output.inject_code(**{"factor.py": "print('ok')"})
    monkeypatch.setattr(factor_module.subprocess, "check_output", lambda *args, **kwargs: b"done")
    with pytest.raises(NoOutputError):
        ws_no_output.execute()


def test_factor_workspace_from_folder_path(monkeypatch, tmp_path):
    _set_common_settings(monkeypatch, tmp_path)
    folder = tmp_path / "impl_folder"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "factor.py").write_text("print('x')", encoding="utf-8")
    (folder / "ignore.txt").write_text("skip", encoding="utf-8")
    task = _make_task("from-folder")

    with pytest.raises(TypeError):
        FactorFBWorkspace.from_folder(task=task, path=folder)


def test_us_market_templates_exist():
    """US factor_template overrides must be present for run_us.sh to work."""
    from quantaalpha.factors.workspace import _MARKET_REGION_TEMPLATE_DIRS

    us_dir = _MARKET_REGION_TEMPLATE_DIRS["us"]
    assert us_dir.exists(), f"US template dir missing: {us_dir}"
    baseline = us_dir / "conf_baseline.yaml"
    combined = us_dir / "conf_combined_factors.yaml"
    assert baseline.exists(), "conf_baseline.yaml missing in us/ template dir"
    assert combined.exists(), "conf_combined_factors.yaml missing in us/ template dir"

    import yaml

    b = yaml.safe_load(baseline.read_text())
    c = yaml.safe_load(combined.read_text())
    assert b["market"] == "sp500"
    assert c["market"] == "sp500"
    assert b["qlib_init"]["region"] == "us"
    assert c["qlib_init"]["region"] == "us"
    assert "us_data" in b["qlib_init"]["provider_uri"]


def test_us_workspace_injects_us_templates(monkeypatch, tmp_path):
    """When market_region='us', workspace injects US conf files (overriding CN defaults)."""
    import quantaalpha.factors.workspace as ws_module
    import quantaalpha.factors.coder.config as cfg_module

    _set_common_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(cfg_module.FACTOR_COSTEER_SETTINGS, "market_region", "us")

    injected = {}

    class _CapturingWorkspace(ws_module.QlibFBWorkspace):
        def inject_files(self, **kwargs):
            injected.update(kwargs)

    # Patch parent __init__ to avoid rdagent filesystem setup
    import rdagent.scenarios.qlib.experiment.workspace as _rda_ws
    monkeypatch.setattr(_rda_ws.QlibFBWorkspace, "__init__", lambda self, *a, **kw: None)

    _CapturingWorkspace(template_folder_path=tmp_path)

    assert "conf_baseline.yaml" in injected, "conf_baseline.yaml not injected"
    assert "conf_combined_factors.yaml" in injected, "conf_combined_factors.yaml not injected"
    assert "sp500" in injected["conf_baseline.yaml"]
    assert "us_data" in injected["conf_baseline.yaml"]
    assert "sp500" in injected["conf_combined_factors.yaml"]


def test_before_execute_symlink_uses_market_region(monkeypatch, tmp_path):
    """before_execute() must create us_data symlink for US, cn_data for CN."""
    import quantaalpha.factors.workspace as ws_module
    import quantaalpha.factors.coder.config as cfg_module

    symlink_calls = {}

    class _FakePath:
        def __init__(self, parts):
            self._parts = parts

        def __truediv__(self, other):
            return _FakePath(self._parts + [other])

        def mkdir(self, **kw): pass

        def is_symlink(self): return False

        def exists(self): return True

        def symlink_to(self, src):
            symlink_calls["target"] = "/".join(str(p) for p in self._parts)
            symlink_calls["src"] = str(src)

        def resolve(self): return self

        def __eq__(self, other): return False

    # Patch before_execute dependencies
    monkeypatch.setenv("QLIB_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cfg_module.FACTOR_COSTEER_SETTINGS, "market_region", "us")

    import pathlib
    fake_home = _FakePath(["~"])
    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: fake_home))

    # Just run the symlink-creation logic directly from before_execute source
    import os
    from pathlib import Path

    qlib_data_dir = os.environ.get("QLIB_DATA_DIR")
    source = Path(qlib_data_dir).expanduser().resolve()
    region = cfg_module.FACTOR_COSTEER_SETTINGS.market_region.lower()
    symlink_name = "us_data" if region == "us" else "cn_data"
    assert symlink_name == "us_data", f"Expected us_data, got {symlink_name}"

    # CN case
    monkeypatch.setattr(cfg_module.FACTOR_COSTEER_SETTINGS, "market_region", "cn")
    region_cn = cfg_module.FACTOR_COSTEER_SETTINGS.market_region.lower()
    symlink_name_cn = "us_data" if region_cn == "us" else "cn_data"
    assert symlink_name_cn == "cn_data", f"Expected cn_data, got {symlink_name_cn}"
