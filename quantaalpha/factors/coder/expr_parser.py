from pyparsing import Word, alphas, alphanums, infix_notation, OpAssoc, one_of, Optional, DelimitedList, Forward, Group
from pyparsing import ParseException
from pyparsing import Regex, Combine, Literal
import sys
import re
import numpy as np

# Use pyparsing packrat for faster nested parsing
from pyparsing import ParserElement
ParserElement.enable_packrat()

sys.setrecursionlimit(5000)

var = (
    Combine(Optional(Literal("$")) + Word(alphas, alphanums + "_"))
).set_name("variable")
# var = Word(alphas, alphanums + "_")

# Number: int/float, optional sign, optional scientific notation
number_pattern = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"
number = Regex(number_pattern)

# Operators
mul_div = one_of("* /", use_regex=True)
add_minus = one_of("+ -")
comparison_op = one_of("> < >= <= == !=")
logical_and = one_of("&& &")
logical_or = one_of("|| |")
conditional_op = ("?", ":")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Flatten nested ParseResults to strings
def flatten_nested_tokens(tokens):
    # import pdb; pdb.set_trace()
    flattened = []
    for token in tokens:
        if isinstance(token, str):
            flattened.append(token)
        elif isinstance(token, list):
            flattened.extend(flatten_nested_tokens(token))
        else:  # ParseResults
            flattened.extend(flatten_nested_tokens(token.asList()))
    return flattened




def parse_arith_op(s, loc, tokens):
    def recursive_build_expression(tokens):
        if len(tokens) == 3:
            A, op, B = tokens
            return build_expression(A, op, B)
        else:
            left = tokens[:-2]
            op = tokens[-2]
            right = tokens[-1]
            left_expr = recursive_build_expression(left)
            return build_expression(left_expr, op, right)
        
    def build_expression(A, op, B):
        A = ''.join(flatten_nested_tokens([A]))
        B = ''.join(flatten_nested_tokens([B]))
        A_is_number = is_number(A)
        B_is_number = is_number(B)
        
        if A_is_number or B_is_number:
            return f"{A}{op}{B}"
        
        else:
            if op == '+':
                return f'ADD({A}, {B})'
                # return f'np.add({A}, {B})'
            elif op == '-':
                return f'SUBTRACT({A}, {B})'
                # return f'np.subtract({A}, {B})'
            elif op == '*':
                return f'MULTIPLY({A}, {B})'
                # return f'np.multiply({A}, {B})'
            elif op == '/':
                return f'DIVIDE({A}, {B})'
                # return f'np.divide({A}, {B})'
            else:
                raise NotImplementedError(f'arith op \'{op}\' is not implemented')
            # if 'BENCHMARKINDEX' in A and 'BENCHMARKINDEX' not in B:
            #     if op == '+':
            #         return f'({B}).add({A}, axis=0)'
            #     elif op == '-':
            #         return f'(-1*{(B)}).add({A}, axis=0)'
            #     elif op == '*':
            #         return f'({B}).mul({A}, axis=0)'
            #     elif op == '/':
            #         return f'(1/{(B)}).mul({A}, axis=0)'
            #     else:
            #         raise NotImplementedError(f'arith op \'{op}\' is not implemented')
            # else:
            #     if op == '+':
            #         return f'({A}).add({B}, axis=0)'
            #     elif op == '-':
            #         return f'({A}).sub({B}, axis=0)'
            #     elif op == '*':
            #         return f'({A}).mul({B}, axis=0)'
            #     elif op == '/':
            #         return f'({A}).div({B}, axis=0)'
            #     else:
            #         raise NotImplementedError(f'arith op \'{op}\' is not implemented')
    
    return recursive_build_expression(tokens[0])

# def parse_arith_op(s, loc, tokens):
#     A = ''.join(flatten_nested_tokens(tokens[0][0]))
#     op = ''.join(flatten_nested_tokens(tokens[0][1]))
#     B = ''.join(flatten_nested_tokens(tokens[0][2]))

#     if A == '' or B == '':
#         raise ParseException(s, loc, f"Operator '{op}' missing operand")
    
#     A_is_number = is_number(A)
#     B_is_number = is_number(B)
    
#     if A_is_number or B_is_number:
#         return f"{A}{op}{B}"
#     else:
#         if 'BENCHMARKINDEX' in A and 'BENCHMARKINDEX' not in B:
#             if op == '+':
#                 return f'({B}).add({A}, axis=0)'
#             elif op == '-':
#                 return f'(-1*{(B)}).add({A}, axis=0)'
#             elif op == '*':
#                 return f'({B}).mul({A}, axis=0)'
#             elif op == '/':
#                 return f'(1/{(B)}).mul({A}, axis=0)'
#             else:
#                 raise NotImplementedError(f'arith op \'{op}\' is not implemented')
#         else:
#             if op == '+':
#                 return f'({A}).add({B}, axis=0)'
#             elif op == '-':
#                 return f'({A}).sub({B}, axis=0)'
#             elif op == '*':
#                 return f'({A}).mul({B}, axis=0)'
#             elif op == '/':
#                 return f'({A}).div({B}, axis=0)'
#             else:
#                 raise NotImplementedError(f'arith op \'{op}\' is not implemented')


def parse_conditional_expression(s, loc, tokens):
    A, B, C = tokens[0][0], tokens[0][2], tokens[0][4]
    A = ''.join(flatten_nested_tokens(A))
    B = ''.join(flatten_nested_tokens(B))
    C = ''.join(flatten_nested_tokens(C))
    return f"WHERE({A}, {B}, {C})"

def parse_comparison_op(s, loc, tokens):
    """Parse comparison op and convert to function call for index alignment."""
    def recursive_build_comparison(tokens):
        if len(tokens) == 3:
            A, op, B = tokens
            return build_comparison(A, op, B)
        else:
            left = tokens[:-2]
            op = tokens[-2]
            right = tokens[-1]
            left_expr = recursive_build_comparison(left)
            return build_comparison(left_expr, op, right)
    
    def build_comparison(A, op, B):
        A = ''.join(flatten_nested_tokens([A]))
        B = ''.join(flatten_nested_tokens([B]))
        A_is_number = is_number(A)
        B_is_number = is_number(B)
        
        if A_is_number and B_is_number:
            return f"{A}{op}{B}"
        
        op_map = {
            '>': 'GT',
            '<': 'LT',
            '>=': 'GE',
            '<=': 'LE',
            '==': 'EQ',
            '!=': 'NE'
        }
        
        func_name = op_map.get(op)
        if func_name:
            return f"{func_name}({A}, {B})"
        else:
            raise NotImplementedError(f"Comparison op '{op}' not implemented")
    
    return recursive_build_comparison(tokens[0])

def parse_logical_expression(s, loc, tokens):
    def recursive_flatten(tokens):
        if len(tokens) == 1:
            return ''.join(flatten_nested_tokens([tokens[0]]))
        else:
            left = tokens[0]
            operator = tokens[1]
            # right = tokens[2]
            left_str = ''.join(flatten_nested_tokens([left]))
            right_str = recursive_flatten(tokens[2:])
            if operator in ["||", "|"]: 
                return f"OR({left_str}, {right_str})"
                # return f"({left_str}) | ({right_str})"
            elif operator in ["&&", "&"]:
                return f"AND({left_str}, {right_str})"
                # return f"({left_str}) & ({right_str})"
    
    return recursive_flatten(tokens[0])


def parse_function_call(s, loc, tokens):
    # unary_operator = tokens[0]
    function_name = tokens[0]
    arguments = tokens[2:-1] 
    # import pdb; pdb.set_trace()


    arguments_flat = []
    # import pdb; pdb.set_trace()
    for arg in arguments:
        if isinstance(arg, str):
            arguments_flat.append(arg)
        else:
            flattened_arg = ''.join(flatten_nested_tokens(arg))
            arguments_flat.append(flattened_arg)
    arguments_str = ','.join(arguments_flat)
    return f"{function_name}({arguments_str})"

expr = Forward()

unary_op = Optional(one_of("+ -")).set_parse_action(lambda t: t[0] if t else '')
function_call = var + '(' + Optional(DelimitedList(expr)) + ')'
function_call.set_parse_action(parse_function_call)
nested_expr = Group('(' + expr + ')')

operand =  Group(unary_op + (function_call | var | number | nested_expr | expr))

# unary_operand = one_of("+ -") + operand
# unary_operand.setParseAction(lambda tokens: ''.join(tokens))
# operand = (unary_operand | function_call | var | number )

def parse_entire_expression(s, loc, tokens):
    # import pdb; pdb.set_trace()
    return ''.join(flatten_nested_tokens(tokens))


def check_for_invalid_operators(expression):
    valid_operators = {"(", ")", ",", "+", "-", "*", "/", "&&", "||", "&", "|", ">", "<", ">=", "<=", "==", "!=", "?", ":", "."}
    pattern = r'([+\-*/,><?:.]{2,})|([><=!&|^`~@#%\\;{}[\]"\'\\]+)'
    found_operators_tuples = re.findall(pattern, expression)
    found_operators = [operator for tup in found_operators_tuples for operator in tup if operator]
    invalid_operators = set(found_operators) - valid_operators
    
    if invalid_operators:
        raise Exception(f"Invalid operator(s): \"{''.join(invalid_operators)}\"")


expr <<= infix_notation(operand, 
    [
        (mul_div, 2, OpAssoc.LEFT, parse_arith_op),
        (add_minus, 2, OpAssoc.LEFT, parse_arith_op),
        (comparison_op, 2, OpAssoc.LEFT, parse_comparison_op),
        (logical_and, 2, OpAssoc.LEFT, parse_logical_expression),
        (logical_or, 2, OpAssoc.LEFT, parse_logical_expression),
        (conditional_op, 3, OpAssoc.RIGHT, parse_conditional_expression)
    ])

    
def check_parentheses_balance(expr):
    if expr.count('(') != expr.count(')'):
        raise ParseException("Unclosed parentheses")

expr.set_parse_action(parse_entire_expression)

def preprocess_unary_minus(factor_expression):
    """Preprocess unary minus: convert -x to (-1 * x) for parser."""
    import re
    
    # "* -(" or "/ -(" -> "* (-1 * (" or "/ (-1 * ("
    factor_expression = re.sub(
        r'(\*\s*)-(\s*\()',
        r'\1(-1 * \2',
        factor_expression
    )
    factor_expression = re.sub(
        r'(/\s*)-(\s*\()',
        r'\1(-1 * \2',
        factor_expression
    )
    
    # "* -func(" or "/ -func("
    factor_expression = re.sub(
        r'(\*\s*)-(\s*[A-Za-z_][A-Za-z0-9_]*\s*\()',
        r'\1(-1 * \2',
        factor_expression
    )
    factor_expression = re.sub(
        r'(/\s*)-(\s*[A-Za-z_][A-Za-z0-9_]*\s*\()',
        r'\1(-1 * \2',
        factor_expression
    )
    
    # "* -$var" or "/ -$var"
    factor_expression = re.sub(
        r'(\*\s*)-(\s*\$[A-Za-z_][A-Za-z0-9_]*)',
        r'\1(-1 * \2)',
        factor_expression
    )
    factor_expression = re.sub(
        r'(/\s*)-(\s*\$[A-Za-z_][A-Za-z0-9_]*)',
        r'\1(-1 * \2)',
        factor_expression
    )
    
    # "+ -(" or "- -("
    factor_expression = re.sub(
        r'(\+\s*)-(\s*\()',
        r'\1(-1 * \2',
        factor_expression
    )
    factor_expression = re.sub(
        r'(-\s*)-(\s*\()',
        r'\1(-1 * \2',
        factor_expression
    )
    
    open_count = factor_expression.count('(')
    close_count = factor_expression.count(')')
    if open_count > close_count:
        factor_expression += ')' * (open_count - close_count)
    
    return factor_expression


def parse_expression(factor_expression):
    if factor_expression is None:
        raise ValueError("Expression is None")
    factor_expression = str(factor_expression).strip()
    if not factor_expression:
        raise ValueError("Expression is empty")

    check_parentheses_balance(factor_expression)
    check_for_invalid_operators(factor_expression)
    
    factor_expression = preprocess_unary_minus(factor_expression)
    try:
        parsed_data_function = expr.parse_string(factor_expression, parse_all=True)[0]
    except RecursionError as e:
        raise ParseException(f"Expression parse recursion error: {e}") from e
    return parsed_data_function



def parse_symbol(expr, columns):
    replace_map = {}
    replace_map.update({
        "TRUE": "True",
        "true": "True",
        "FALSE": "False",
        "false": "False",
        "NAN": "np.nan",
        "NaN": "np.nan",
        "nan": "np.nan",
        "NULL": "np.nan",
        "null": "np.nan"
    })
    for col in columns:
        replace_map.update({col: col.replace('$', '')})
        # replace_map.update({col.replace('$', '').upper(): col.replace('$', '')})

    for var, var_df in sorted(replace_map.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = rf"(?<![A-Za-z0-9_\$]){re.escape(var)}(?![A-Za-z0-9_])"
        expr = re.sub(pattern, var_df, expr)
    return expr

if __name__ == '__main__':
    parse_expression("RANK(DELTA($open, 1) - DELTA($open, 1)) / (1e-8 + 1)")
