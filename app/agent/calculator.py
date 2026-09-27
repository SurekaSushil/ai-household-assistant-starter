import ast
import operator

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_MAX_EXPRESSION_LENGTH = 200
_MAX_ABS_RESULT = 1e100
_MAX_ABS_EXPONENT = 1_000


def calculate(expression: str) -> float:
    """Evaluate a small arithmetic expression. Does not use eval()."""

    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long")
    node = ast.parse(expression, mode="eval")

    def visit(item: ast.AST) -> int | float:
        if isinstance(item, ast.Expression):
            return visit(item.body)
        if isinstance(item, ast.Constant) and _is_number(item.value):
            return item.value
        if isinstance(item, ast.BinOp) and type(item.op) in _ALLOWED_BINOPS:
            left = visit(item.left)
            right = visit(item.right)
            if isinstance(item.op, ast.Pow) and abs(right) > _MAX_ABS_EXPONENT:
                raise ValueError("Exponent is too large")
            return _ALLOWED_BINOPS[type(item.op)](left, right)
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
            value = visit(item.operand)
            return value if isinstance(item.op, ast.UAdd) else -value
        raise ValueError("Unsupported expression")

    result = visit(node)
    if abs(result) > _MAX_ABS_RESULT:
        raise ValueError("Result is too large")
    return float(result)


def _is_number(value: object) -> bool:
    # bool is a subclass of int; True/False are not arithmetic literals.
    return isinstance(value, (int, float)) and not isinstance(value, bool)
