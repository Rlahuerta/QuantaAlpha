from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import quantaalpha.factors.coder as factor_coder_module
from quantaalpha.factors.coder import FactorCoSTEER, FactorCoder, FactorParser
from quantaalpha.factors.coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy,
    FactorParsingStrategy,
    FactorRunningStrategy,
)
from quantaalpha.core.scenario import Scenario


class _Scenario(Scenario):
    @property
    def background(self) -> str:
        return "bg"

    @property
    def interface(self) -> str:
        return "itf"

    @property
    def output_format(self) -> str:
        return "out"

    @property
    def simulator(self) -> str:
        return "sim"

    @property
    def rich_style_description(self) -> str:
        return "rich"

    def get_scenario_all_desc(self, task=None, filtered_tag=None, simple_background=None) -> str:  # noqa: ANN001, ANN201
        return "desc"


def test_factor_coder_entrypoints_wire_expected_strategy(monkeypatch):
    captured = []

    def _fake_costeer_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN201
        captured.append(kwargs)

    monkeypatch.setattr(factor_coder_module.CoSTEER, "__init__", _fake_costeer_init)
    scen = _Scenario()

    FactorCoSTEER(scen)
    FactorParser(scen)
    FactorCoder(scen)

    assert captured[0]["evolving_version"] == 2
    assert isinstance(captured[0]["es"], FactorMultiProcessEvolvingStrategy)
    assert isinstance(captured[1]["es"], FactorParsingStrategy)
    assert isinstance(captured[2]["es"], FactorRunningStrategy)


def test_regulator_init_importerror_fallback_branch(monkeypatch):
    regulator_init_path = Path(__file__).resolve().parents[2] / "quantaalpha" / "factors" / "regulator" / "__init__.py"
    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, ANN201
        if name == "quantaalpha.factors.regulator.consistency_checker":
            raise ImportError("forced import failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    spec = importlib.util.spec_from_file_location("regulator_init_forced_import_error", regulator_init_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    assert module.CONSISTENCY_CHECKER_AVAILABLE is False
    assert module.FactorConsistencyChecker is None
    assert module.ConsistencyCheckResult is None
