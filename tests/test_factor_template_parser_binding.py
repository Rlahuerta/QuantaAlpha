from pathlib import Path

import numpy as np
import pandas as pd


TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "quantaalpha"
    / "factors"
    / "coder"
    / "template.jinjia2"
)


def _write_daily_data(tmp_path, columns):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )
    base = np.arange(len(idx), dtype=float)
    frame = pd.DataFrame({col: base + i + 1 for i, col in enumerate(columns)}, index=idx)
    frame.to_hdf(tmp_path / "daily_pv.h5", key="data")


def _run_template_calculation(tmp_path, expression):
    namespace = {"__name__": "quantaalpha_template_test"}
    code = compile(TEMPLATE_PATH.read_text(encoding="utf-8"), str(TEMPLATE_PATH), "exec")
    exec(code, namespace)
    namespace["calculate_factor"](expression, "alpha_test")
    return pd.read_hdf(tmp_path / "result.h5", key="data")


def test_template_calculation_supports_plain_columns(tmp_path, monkeypatch):
    _write_daily_data(tmp_path, ["return", "volume"])
    monkeypatch.chdir(tmp_path)

    result = _run_template_calculation(
        tmp_path,
        "RANK(TS_SUM($return, 2) * TS_SUM($volume, 2))",
    )

    assert len(result) == 8
    assert result.notna().any()


def test_template_calculation_supports_dollar_prefixed_columns(tmp_path, monkeypatch):
    _write_daily_data(tmp_path, ["$return", "$volume"])
    monkeypatch.chdir(tmp_path)

    result = _run_template_calculation(
        tmp_path,
        "RANK(TS_SUM($return, 2) * TS_SUM($volume, 2))",
    )

    assert len(result) == 8
    assert result.notna().any()
