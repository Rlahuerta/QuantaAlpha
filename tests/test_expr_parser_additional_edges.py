import pytest
from pyparsing import ParseException

import quantaalpha.factors.coder.expr_parser as expr_parser_module
from quantaalpha.factors.coder.expr_parser import (
    check_for_invalid_operators,
    check_parentheses_balance,
    flatten_nested_tokens,
    parse_expression,
    parse_symbol,
    preprocess_unary_minus,
)


def test_flatten_and_symbol_replacement_edges():
    nested = ["A", ["B", ["C", "D"]], ["E"]]
    assert flatten_nested_tokens(nested) == ["A", "B", "C", "D", "E"]

    expr = "TRUE && false && NAN && null && $return + $return_rate"
    parsed = parse_symbol(expr, ["$return"])
    assert "True" in parsed
    assert "np.nan" in parsed
    assert "return + $return_rate" in parsed


def test_preprocess_unary_minus_and_parse_expression_complex_cases():
    preprocessed = preprocess_unary_minus("$close * -$open + -($high)")
    assert "(-1 * $open)" in preprocessed
    assert preprocessed.count("(") == preprocessed.count(")")

    parsed = parse_expression("($close > $open) ? $close : $open")
    assert "WHERE(" in parsed
    assert "GT(" in parsed

    parsed_logic = parse_expression("($close > $open) && ($high >= $low)")
    assert "AND(" in parsed_logic
    assert "GE(" in parsed_logic

    parsed_compare_numbers = parse_expression("1 < 2")
    assert parsed_compare_numbers == "1<2"


def test_invalid_operator_and_parentheses_errors():
    with pytest.raises(Exception, match="Invalid operator"):
        check_for_invalid_operators("$close ^^ $open")

    with pytest.raises(ParseException, match="Unclosed parentheses"):
        check_parentheses_balance("($close + $open")

    with pytest.raises(ParseException):
        parse_expression("($close + $open")


def test_parse_expression_recursion_error_path(monkeypatch):
    original_parse_string = expr_parser_module.expr.parseString

    def _raise_recursion(*args, **kwargs):
        raise RecursionError("deep recursion")

    monkeypatch.setattr(expr_parser_module.expr, "parseString", _raise_recursion)
    with pytest.raises(ParseException, match="recursion error"):
        parse_expression("$close + $open")
    monkeypatch.setattr(expr_parser_module.expr, "parseString", original_parse_string)
