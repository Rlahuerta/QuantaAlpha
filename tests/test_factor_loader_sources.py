import json

import pytest

from quantaalpha.backtest.factor_loader import FactorLoader


def _write_factor_library(tmp_path, factors):
    factor_file = tmp_path / 'factor_library.json'
    factor_file.write_text(json.dumps({'factors': factors}), encoding='utf-8')
    return str(factor_file)


def test_custom_factor_loader_applies_quality_filter(tmp_path):
    factor_file = _write_factor_library(
        tmp_path,
        {
            'f1': {
                'factor_name': 'HighQualityA',
                'factor_expression': 'TS_MEAN($close, 5)',
                'quality': 'high',
            },
            'f2': {
                'factor_name': 'LowQuality',
                'factor_expression': 'TS_MEAN($close, 10)',
                'quality': 'low',
            },
            'f3': {
                'factor_name': 'HighQualityWithCache',
                'factor_expression': 'TS_STD($close, 10)',
                'quality': 'high',
                'cache_location': '/tmp/cache.pkl',
            },
            'f4': {
                'factor_name': 'NoExpr',
                'factor_expression': '',
                'quality': 'high',
            },
        },
    )
    loader = FactorLoader(
        {
            'factor_source': {
                'type': 'custom',
                'custom': {
                    'json_files': [factor_file],
                    'quality_filter': 'high',
                },
            }
        }
    )

    qlib_compatible, custom_factors = loader.load_factors()

    assert qlib_compatible == {}
    assert {item['factor_name'] for item in custom_factors} == {
        'HighQualityA',
        'HighQualityWithCache',
    }
    cached = next(item for item in custom_factors if item['factor_name'] == 'HighQualityWithCache')
    assert cached['cache_location'] == '/tmp/cache.pkl'


def test_combined_factor_loader_merges_official_and_custom(tmp_path):
    factor_file = _write_factor_library(
        tmp_path,
        {
            'f1': {
                'factor_name': 'CustomMomentum',
                'factor_expression': 'RANK($close)',
                'quality': 'high',
            }
        },
    )
    loader = FactorLoader(
        {
            'factor_source': {
                'type': 'combined',
                'combined': {
                    'official_source': 'alpha158_20',
                    'include_custom': True,
                },
                'custom': {
                    'json_files': [factor_file],
                },
            }
        }
    )

    qlib_compatible, needs_llm = loader.load_factors()

    assert len(qlib_compatible) == len(FactorLoader.ALPHA158_20_FACTORS)
    assert len(needs_llm) == 1
    assert needs_llm[0]['factor_name'] == 'CustomMomentum'


def test_factor_loader_rejects_unknown_source():
    loader = FactorLoader({'factor_source': {'type': 'unsupported'}})

    with pytest.raises(ValueError, match='Unsupported factor source type'):
        loader.load_factors()


def test_parse_factor_json_splits_qlib_compatible_and_needs_llm(tmp_path):
    factor_file = _write_factor_library(
        tmp_path,
        {
            "f1": {
                "factor_name": "Compat",
                "factor_expression": "TS_MEAN($close, 5)",
            },
            "f2": {
                "factor_name": "NeedsLLM",
                "factor_expression": "RANK($close)",
            },
        },
    )
    loader = FactorLoader({"factor_source": {"type": "custom"}})

    qlib_compatible, needs_llm = loader._parse_factor_json(file_path=factor_file)

    assert qlib_compatible["Compat"] == "Mean($close, 5)"
    assert needs_llm[0]["factor_name"] == "NeedsLLM"


def test_convert_to_qlib_expression_returns_none_for_unsupported_patterns():
    loader = FactorLoader({"factor_source": {"type": "custom"}})

    assert loader._convert_to_qlib_expression("TS_MEAN($close, 5) + RANK($close)") is None


def test_get_factor_info_variants(tmp_path):
    custom_file = _write_factor_library(tmp_path, {})

    alpha_info = FactorLoader({"factor_source": {"type": "alpha360"}}).get_factor_info()
    custom_info = FactorLoader(
        {"factor_source": {"type": "custom", "custom": {"json_files": [custom_file]}}}
    ).get_factor_info()
    unknown_info = FactorLoader({"factor_source": {"type": "x"}}).get_factor_info()

    assert alpha_info["type"] == "alpha360"
    assert alpha_info["count"] == "dynamic"
    assert custom_info["json_files"] == [custom_file]
    assert unknown_info["description"] == "Unknown factor source"
