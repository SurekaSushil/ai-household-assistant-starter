import pytest

from app.agent.calculator import calculate

UNSAFE_EXPRESSIONS = [
    '__import__("os").system("dir")',
    "len(1)",
    "(1).real",
    "os",
    "[1, 2]",
    "'hello'",
    '"1+1"',
    "True + 1",
]


def test_multiply_then_add() -> None:
    assert calculate("(37 * 19) + 8") == 711.0


def test_power() -> None:
    assert calculate("2 ** 10") == 1024.0


def test_unary_minus() -> None:
    assert calculate("-(3 + 4)") == -7.0


def test_rejects_overlong_expression() -> None:
    with pytest.raises(ValueError, match="too long"):
        calculate("1+" * 101)


@pytest.mark.parametrize("expression", UNSAFE_EXPRESSIONS)
def test_rejects_unsupported_syntax(expression: str) -> None:
    with pytest.raises(ValueError, match="Unsupported expression"):
        calculate(expression)


def test_rejects_function_calls() -> None:
    with pytest.raises(ValueError, match="Unsupported expression"):
        calculate("abs(-1)")


def test_rejects_attribute_access() -> None:
    with pytest.raises(ValueError, match="Unsupported expression"):
        calculate("(1).bit_length()")
