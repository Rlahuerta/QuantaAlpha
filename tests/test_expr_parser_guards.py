import pytest

from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol


@pytest.mark.parametrize("bad_expr", [None, "", "   "])
def test_parse_expression_rejects_empty_inputs(bad_expr):
    with pytest.raises(ValueError):
        parse_expression(bad_expr)


def test_parse_expression_parses_valid_expression():
    parsed = parse_expression("RANK(TS_MEAN($return, 20) * TS_CORR($return, $volume, 20))")
    assert "RANK(" in parsed
    assert "TS_MEAN(" in parsed
    assert "TS_CORR(" in parsed


def test_parse_symbol_avoids_partial_token_replacement():
    expr = "TS_MEAN($return, 20) + TS_MEAN($return_rate, 5)"
    parsed = parse_symbol(expr, ["$return"])
    assert "TS_MEAN(return, 20)" in parsed
    assert "$return_rate" in parsed

