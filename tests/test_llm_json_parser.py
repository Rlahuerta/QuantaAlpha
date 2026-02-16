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

