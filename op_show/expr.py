"""Small integer expression language for launch and tile formulas.

Expressions are data, never Python code. No attributes, subscripts, calls other
than ceildiv/min/max, or floating point operations are accepted.
"""

import ast


class ExpressionError(ValueError):
    pass


def evaluate(expression, variables):
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError("formula must be a nonempty string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid formula {expression!r}") from exc

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ExpressionError(f"unknown variable {node.id!r} in {expression!r}")
            value = variables[node.id]
            if type(value) is not int:
                raise ExpressionError(f"variable {node.id!r} is not an integer")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                if right == 0:
                    raise ExpressionError("division by zero")
                return left // right if isinstance(node.op, ast.FloorDiv) else left % right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [walk(arg) for arg in node.args]
            if node.func.id == "ceildiv" and len(args) == 2:
                if args[1] <= 0:
                    raise ExpressionError("ceildiv divisor must be positive")
                return -(-args[0] // args[1])
            if node.func.id in ("min", "max") and len(args) >= 1:
                return min(args) if node.func.id == "min" else max(args)
        raise ExpressionError(f"unsupported formula syntax in {expression!r}")

    value = walk(tree)
    if abs(value) > 10**12:
        raise ExpressionError("formula result exceeds supported range")
    return value
