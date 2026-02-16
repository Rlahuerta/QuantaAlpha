import pandas as pd
import pytest

import quantaalpha.factors.coder.factor_ast as ast_module


def test_node_string_and_tree_views(capsys):
    var_node = ast_module.VarNode("$close")
    num_node = ast_module.NumberNode(1.5)
    fn_node = ast_module.FunctionNode("TS_MEAN", [var_node, num_node])
    bin_node = ast_module.BinaryOpNode("+", var_node, num_node)
    cond_node = ast_module.ConditionalNode(var_node, num_node, var_node)
    unary_node = ast_module.UnaryOpNode("-", var_node)

    assert str(var_node) == "$close"
    assert "VAR($close)" in var_node.tree_str()
    assert str(num_node) == "1.5"
    assert "NUM(1.5)" in num_node.tree_str()
    assert "FUNC(TS_MEAN)" in fn_node.tree_str()
    assert "OP(+)" in bin_node.tree_str()
    assert "CONDITIONAL" in cond_node.tree_str()
    assert "UNARY(-)" in unary_node.tree_str()

    var_node.print_tree()
    captured = capsys.readouterr()
    assert "VAR($close)" in captured.out


def test_parse_expression_valid_and_invalid_paths():
    parsed_add = ast_module.parse_expression("TS_MEAN($close, 5) + $open")
    parsed_unary = ast_module.parse_expression("-$open")
    parsed_cond = ast_module.parse_expression("$close > 1 ? $open : $low")

    assert isinstance(parsed_add, ast_module.BinaryOpNode)
    assert isinstance(parsed_unary, ast_module.UnaryOpNode)
    assert isinstance(parsed_cond, ast_module.ConditionalNode)

    with pytest.raises(ValueError, match="Failed to parse expression"):
        ast_module.parse_expression("TS_MEAN(")


def test_create_node_helpers_and_node_equality_paths():
    var = ast_module.create_var_node(["$high"])
    num = ast_module.create_number_node(["2"])
    fn = ast_module.create_function_node(["TS_SUM", "(", ast_module.VarNode("$close"), ")"])
    bin_node = ast_module.create_binary_op_node([[ast_module.VarNode("$a"), "+", ast_module.VarNode("$b")]])
    cond_node = ast_module.create_conditional_node(
        [[ast_module.VarNode("$x"), "?", ast_module.VarNode("$y"), ":", ast_module.VarNode("$z")]]
    )
    unary_node = ast_module.create_unary_op_node([["-", ast_module.VarNode("$x")]])

    assert isinstance(var, ast_module.VarNode)
    assert isinstance(num, ast_module.NumberNode)
    assert isinstance(fn, ast_module.FunctionNode)
    assert isinstance(bin_node, ast_module.BinaryOpNode)
    assert isinstance(cond_node, ast_module.ConditionalNode)
    assert isinstance(unary_node, ast_module.UnaryOpNode)

    assert ast_module.are_nodes_equal(ast_module.NumberNode(1), ast_module.NumberNode(1))
    assert not ast_module.are_nodes_equal(ast_module.NumberNode(1), ast_module.NumberNode(2))
    assert ast_module.are_nodes_equal(ast_module.VarNode("$a"), ast_module.VarNode("$a"))
    assert not ast_module.are_nodes_equal(ast_module.VarNode("$a"), ast_module.VarNode("$b"))
    assert ast_module.are_nodes_equal(
        ast_module.FunctionNode("F", [ast_module.VarNode("$a")]),
        ast_module.FunctionNode("F", [ast_module.VarNode("$b")]),
    )
    assert not ast_module.are_nodes_equal(
        ast_module.FunctionNode("F", [ast_module.VarNode("$a")]),
        ast_module.FunctionNode("G", [ast_module.VarNode("$a")]),
    )
    assert ast_module.are_nodes_equal(
        ast_module.BinaryOpNode("+", ast_module.VarNode("$a"), ast_module.VarNode("$b")),
        ast_module.BinaryOpNode("+", ast_module.VarNode("$x"), ast_module.VarNode("$y")),
    )
    assert ast_module.are_nodes_equal(
        ast_module.ConditionalNode(ast_module.VarNode("$a"), ast_module.VarNode("$b"), ast_module.VarNode("$c")),
        ast_module.ConditionalNode(ast_module.VarNode("$x"), ast_module.VarNode("$y"), ast_module.VarNode("$z")),
    )
    assert not ast_module.are_nodes_equal(ast_module.VarNode("$a"), ast_module.NumberNode(1))


def test_find_largest_common_subtree_and_compare_expressions():
    commutative_match = ast_module.compare_expressions("$a + $b", "$b + $a")
    assert commutative_match is not None
    assert commutative_match.size >= 3

    non_commutative_match = ast_module.compare_expressions("$a - $b", "$b - $a")
    assert non_commutative_match is not None
    assert non_commutative_match.size >= 1

    cond_match = ast_module.compare_expressions("$x > 1 ? $y : $z", "$x > 1 ? $y : $z")
    assert cond_match is not None
    assert cond_match.size >= 5


def test_subtree_match_str_has_known_name_error():
    match = ast_module.SubtreeMatch(ast_module.VarNode("$a"), ast_module.VarNode("$a"), 1)
    with pytest.raises(NameError):
        str(match)


def test_match_alphazoo_selection_and_error_handling(capsys, monkeypatch):
    class _FakeDF:
        def iterrows(self):
            yield 0, ("f1", "$a + 1")
            yield 1, ("f2", "$a + 2")
            yield 2, ("f3", "BAD_EXPR")

    def _fake_compare(prop_expr, alpha_expr):
        if alpha_expr == "BAD_EXPR":
            raise RuntimeError("boom")
        size = 2 if alpha_expr.endswith("1") else 3
        return ast_module.SubtreeMatch(ast_module.VarNode("$a"), ast_module.VarNode("$a"), size)

    monkeypatch.setattr(ast_module, "compare_expressions", _fake_compare)
    max_size, matched_subtree, matched_alpha = ast_module.match_alphazoo("$a + 3", _FakeDF())

    assert max_size == 3
    assert isinstance(matched_subtree, ast_module.VarNode)
    assert matched_alpha == "$a + 2"
    captured = capsys.readouterr()
    assert "Error comparing alpha" in captured.out


def test_count_helpers_cover_number_variable_and_feature_paths():
    expr = "(($close - TS_MIN($low, 14)) / (TS_MAX($high, 14) - TS_MIN($low, 14) + 1e-8)) * 100"
    assert ast_module.count_free_args(expr) == 5
    assert ast_module.count_unique_vars(expr) == 3
    assert ast_module.count_base_features(expr) == 3
    assert ast_module.count_all_nodes(expr) > 0
    assert ast_module.calculate_symbol_length(f"  {expr}  ") == len(expr)

    tree = ast_module.parse_expression("$a > 1 ? TS_MEAN($b, 3) : -$c")
    unique_vars = set()
    base_features = set()
    ast_module.collect_unique_vars(tree, unique_vars)
    ast_module.collect_base_features(tree, base_features)
    # Current collector implementation does not descend into UnaryOpNode.
    assert unique_vars == {"$a", "$b"}
    assert base_features == {"$a", "$b"}

    assert ast_module.count_number_nodes(tree) == 2
    assert ast_module.count_nodes(tree) >= 1


def test_count_nodes_fallback_for_unknown_node_type():
    class _UnknownNode(ast_module.Node):
        pass

    unknown = _UnknownNode()
    assert ast_module.count_number_nodes(unknown) == 0
    assert ast_module.count_nodes(unknown) == 0
