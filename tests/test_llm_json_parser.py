import json

import pytest

from quantaalpha.llm.client import robust_json_parse


@pytest.mark.parametrize(
    "payload, expected",
    [
        (
            """```json
"final_feedback": "ok",
"final_decision": false
```""",
            {"final_feedback": "ok", "final_decision": False},
        ),
        (
            """{
description: "x",
variables: {"a": 1,},
expression: "TS_MEAN($return, 20)",
}""",
            {"description": "x", "expression": "TS_MEAN($return, 20)"},
        ),
    ],
)
def test_robust_json_parse_handles_malformed_payloads(payload, expected):
    parsed = robust_json_parse(payload)
    for key, value in expected.items():
        assert parsed[key] == value


def test_robust_json_parse_raises_for_unstructured_text():
    with pytest.raises(json.JSONDecodeError):
        robust_json_parse("This is not JSON output.")


def test_robust_json_parse_handles_markdown_json_block_direct():
    parsed = robust_json_parse("```json\n{\"alpha\": 1}\n```")
    assert parsed == {"alpha": 1}


def test_robust_json_parse_handles_markdown_json_block_after_fix():
    parsed = robust_json_parse("```json\n{\"alpha\": 1,}\n```")
    assert parsed == {"alpha": 1}


def test_robust_json_parse_handles_unbraced_key_value_block():
    parsed = robust_json_parse("alpha: 1,\nbeta: 2,")
    assert parsed["alpha"].startswith("1")
    assert parsed["beta"].startswith("2")


def test_robust_json_parse_handles_escaped_text_and_trailing_suffix():
    payload = 'prefix {"msg":"a\\\"b","value":2} trailing text'
    parsed = robust_json_parse(payload)
    assert parsed == {"msg": 'a"b', "value": 2}


def test_robust_json_parse_loose_extraction_fallback():
    payload = 'broken prefix {invalid: [} middle {"k": 3} suffix'
    parsed = robust_json_parse(payload)
    assert parsed == {"k": 3}


def test_robust_json_parse_empty_markdown_block_raises():
    with pytest.raises(json.JSONDecodeError):
        robust_json_parse("```json\n\n```")
