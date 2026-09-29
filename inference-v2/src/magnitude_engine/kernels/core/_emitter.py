"""Private emission of captured scalar expressions and checked tensor index maps.

GPU algorithms, reductions and memory protocols are authored in Metal.
"""

from dataclasses import dataclass

from .plan import identifier


class Expression:
    def __bool__(self):
        raise TypeError("device expressions cannot control Python execution")


@dataclass(frozen=True)
class Symbol(Expression):
    name: str

    def __post_init__(self):

        identifier(self.name)
        if self.name in {
            "thread",
            "threadgroup",
            "device",
            "constant",
            "kernel",
            "float",
            "uint",
            "int",
            "bool",
            "auto",
            "if",
            "else",
            "for",
            "while",
            "return",
        }:
            raise ValueError(f"reserved Metal identifier: {self.name}")


@dataclass(frozen=True)
class Literal(Expression):
    value: int | float | bool


@dataclass(frozen=True)
class Binary(Expression):
    operator: str
    left: Expression
    right: Expression

    def __post_init__(self):
        if self.operator not in {
            "+",
            "-",
            "*",
            "/",
            "%",
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
            "&&",
            "||",
            "&",
            "|",
            "^",
            "<<",
            ">>",
        }:
            raise ValueError(f"unsupported binary operator: {self.operator}")


@dataclass(frozen=True)
class Negative(Expression):
    value: Expression


@dataclass(frozen=True)
class Call(Expression):
    function: str
    arguments: tuple[Expression, ...]
    template: tuple[str, ...] = ()


@dataclass(frozen=True)
class Cast(Expression):
    dtype: str
    value: Expression


@dataclass(frozen=True)
class Index(Expression):
    array: Expression
    index: Expression


@dataclass(frozen=True)
class Member(Expression):
    value: Expression
    member: str


@dataclass(frozen=True)
class Select(Expression):
    condition: Expression
    yes: Expression
    no: Expression


class Statement:
    pass


@dataclass(frozen=True)
class Assign(Statement):
    target: Expression
    value: Expression


@dataclass(frozen=True)
class If(Statement):
    condition: Expression
    yes: tuple[Statement, ...]
    no: tuple[Statement, ...] = ()


@dataclass(frozen=True)
class For(Statement):
    variable: Symbol
    start: Expression
    stop: Expression
    body: tuple[Statement, ...]
    step: Expression = Literal(1)
    unroll: bool = False


def expression(value: Expression) -> str:
    match value:
        case Symbol(name):
            return name
        case Literal(x):
            return str(int(x)) if isinstance(x, (bool, int)) else f"{x!r}f"
        case Negative(x):
            return f"(-{expression(x)})"
        case Binary(op, left, right):
            return f"({expression(left)} {op} {expression(right)})"
        case Call(name, args, template):
            suffix = f"<{', '.join(template)}>" if template else ""
            return f"{name}{suffix}({', '.join(map(expression, args))})"
        case Cast(dtype, x):
            return f"{dtype}({expression(x)})"
        case Index(x, index):
            return f"{expression(x)}[{expression(index)}]"
        case Member(x, member):
            return f"{expression(x)}.{member}"
        case Select(test, yes, no):
            return f"({expression(test)} ? {expression(yes)} : {expression(no)})"
        case _:
            raise TypeError(f"unrecognized device expression: {type(value).__name__}")


@dataclass(frozen=True)
class Let(Statement):
    symbol: Symbol
    dtype: str
    value: Expression


def emit(statements: tuple[Statement, ...], depth: int = 0) -> str:
    lines = []

    def line(text):
        lines.append("    " * depth + text)

    for statement in statements:
        match statement:
            case Let(symbol, dtype, value):
                line(f"{dtype} {symbol.name} = {expression(value)};")
            case Assign(target, value):
                line(f"{expression(target)} = {expression(value)};")
            case If(condition, yes, no):
                line(f"if ({expression(condition)}) {{")
                lines.append(emit(yes, depth + 1))
                if no:
                    line("} else {")
                    lines.append(emit(no, depth + 1))
                line("}")
            case For(variable, start, stop, body, step, unroll):
                if unroll:
                    line("#pragma clang loop unroll(full)")
                line(
                    f"for (uint {variable.name} = {expression(start)}; "
                    f"{variable.name} < {expression(stop)}; "
                    f"{variable.name} += {expression(step)}) {{"
                )
                lines.append(emit(body, depth + 1))
                line("}")
            case _:
                raise TypeError(
                    f"unsupported generated scalar statement: {type(statement).__name__}"
                )
    return "\n".join(lines)
