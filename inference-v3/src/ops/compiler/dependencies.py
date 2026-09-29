"""Loaded Python definition identity, independent of unrelated file edits.

This inspects authored Python dependencies; it neither generates Python/TileLang
source nor edits compiler IR. Identity describes loaded code, not whatever bytes
may happen to have replaced its source file since the module was imported.
"""

from __future__ import annotations

import dis
import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from types import CodeType, FunctionType, ModuleType


@dataclass(frozen=True, slots=True)
class CodeDependency:
    module: str
    symbol: str
    fingerprint: str


def _constant(value):
    if isinstance(value, CodeType):
        return _code(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", value.real.hex(), value.imag.hex())
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, (tuple, frozenset)):
        items = tuple(_constant(item) for item in value)
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True)) if isinstance(value, frozenset) else items
    if value is Ellipsis:
        return ("ellipsis",)
    raise TypeError(f"unsupported authored constant {type(value).__qualname__}")


@lru_cache(maxsize=2048)
def _code(code: CodeType):
    # Filenames, line tables and starting lines are diagnostics, not executable
    # meaning. Adding an unrelated function above this one must not evict it.
    return (code.co_code.hex(), code.co_exceptiontable.hex(), code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags,
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
            tuple(_constant(value) for value in code.co_consts))


@lru_cache(maxsize=2048)
def _nested(code: CodeType):
    nested = [code]
    for value in code.co_consts:
        if isinstance(value, CodeType):
            nested.extend(_nested(value))
    return tuple(nested)


@lru_cache(maxsize=2048)
def _names(code: CodeType):
    return frozenset(name for nested in _nested(code) for name in nested.co_names)


@lru_cache(maxsize=2048)
def _instructions(code: CodeType):
    # Code objects are immutable. These bounded caches survive source refresh;
    # resolved imports, globals and definition dependencies deliberately do not.
    return tuple(dis.get_instructions(code))


def _imports(function: FunctionType):
    """Resolve ordinary function-local imports from their Python bytecode."""
    for code in _nested(function.__code__):
        instructions = _instructions(code)
        module = None
        scanned = set()
        for index, instruction in enumerate(instructions):
            if instruction.opname == "IMPORT_NAME":
                level = instructions[index - 2].argval
                name = instruction.argval
                if not isinstance(level, int) or not isinstance(name, str):
                    raise TypeError("cannot identify an authored operation import")
                if level:
                    name = importlib.util.resolve_name("." * level + name, function.__globals__["__package__"])
                module = importlib.import_module(name)
            elif instruction.opname == "IMPORT_FROM" and module is not None:
                value = getattr(module, instruction.argval, None)
                if value is not None:
                    yield f"{module.__name__}.{instruction.argval}", value
            elif (instruction.opname in ("STORE_FAST", "STORE_NAME") and module is not None
                  and module.__name__ not in scanned):
                # Plain module imports can be followed by attribute accesses.
                # Scan each imported namespace once, not once for every later
                # local assignment. Include attributes used by nested closures.
                scanned.add(module.__name__)
                for name in sorted(_names(function.__code__)):
                    value = vars(module).get(name)
                    if value is not None:
                        yield f"{module.__name__}.{name}", value


def code_dependencies(value) -> tuple[CodeDependency, ...]:
    """Transitive authored functions/macros, including helpers and local imports.

    External library implementations are identified separately by compiler/runtime
    provenance. Only ops-owned Python is traversed here, avoiding traversal of an
    entire NumPy/TileLang interpreter for one macro call.
    """
    if isinstance(value, (type, FunctionType)):
        return _definition_dependencies(value)
    original = getattr(value, "orig_func", None)
    if isinstance(original, FunctionType):
        return _definition_dependencies(original)
    # Bound bodies can carry macros/closures in fields. Their class alone does
    # not describe the code that will execute (notably OperationContext.kernel).
    return _collect_dependencies(value)


def invalidate_code_cache() -> None:
    """Call only after installing a new loaded source revision."""
    _definition_dependencies.cache_clear()


@lru_cache(maxsize=1024)
def _definition_dependencies(value) -> tuple[CodeDependency, ...]:
    return _collect_dependencies(value)


def _collect_dependencies(value) -> tuple[CodeDependency, ...]:
    found: dict[tuple[str, str], CodeDependency] = {}
    visited: set[int] = set()
    authored_module = getattr(value, "__module__", type(value).__module__)

    def authored(module):
        return module == authored_module or module.startswith("ops.")

    def visit(item, symbol=None):
        if id(item) in visited:
            return
        visited.add(id(item))
        original = getattr(item, "orig_func", None)
        if isinstance(original, FunctionType):
            visit(original)
            return
        if isinstance(item, FunctionType):
            if not authored(item.__module__):
                return
            owner_symbol = symbol or item.__qualname__
            constants = []
            names = _names(item.__code__)
            for name in sorted(names):
                dependency = item.__globals__.get(name)
                if isinstance(dependency, (str, int, float, bool, bytes, tuple, frozenset)):
                    try:
                        constants.append((name, _constant(dependency)))
                    except TypeError:
                        visit(dependency)
                elif isinstance(dependency, Enum):
                    constants.append((name, dependency.value))
                elif isinstance(dependency, ModuleType):
                    for attribute in names:
                        visit(vars(dependency).get(attribute))
                else:
                    visit(dependency)
            for index, cell in enumerate(item.__closure__ or ()):
                captured = cell.cell_contents
                # Dataclass repr wrappers capture generated functions with the
                # same __create_fn__ qualname across unrelated classes. Their
                # identity is scoped to the owning method, not a module global.
                generated = isinstance(captured, FunctionType) and "__create_fn__" in captured.__qualname__
                visit(captured, f"{owner_symbol}.<closure:{index}>" if generated else None)
            for name, dependency in _imports(item):
                try:
                    constants.append((f"import:{name}", _constant(dependency)))
                except TypeError:
                    if isinstance(dependency, Enum):
                        constants.append((f"import:{name}", dependency.value))
                    else:
                        visit(dependency)
            defaults = []
            for name, default in (
                *enumerate(item.__defaults__ or ()), *(item.__kwdefaults__ or {}).items(),
            ):
                try:
                    defaults.append((str(name), _constant(default)))
                except TypeError:
                    visit(default)
                    defaults.append((str(name), type(default).__module__, type(default).__qualname__))
            payload = (_code(item.__code__), constants, defaults)
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            key = item.__module__, owner_symbol
            record = CodeDependency(*key, fingerprint)
            if key in found and found[key] != record:
                raise ValueError(f"mixed loaded revisions of {key[0]}.{key[1]}")
            found[key] = record
        elif isinstance(item, type):
            if authored(item.__module__):
                for name, member in vars(item).items():
                    member = (member.__func__ if isinstance(member, (staticmethod, classmethod)) else
                              member.fget if isinstance(member, property) else member)
                    visit(member, f"{item.__qualname__}.{name}" if isinstance(member, FunctionType) else None)
        elif isinstance(item, Mapping):
            for member in item.values():
                visit(member)
        elif isinstance(item, (tuple, list, set, frozenset)):
            for member in item:
                visit(member)
        elif callable(item) and authored(type(item).__module__):
            visit(type(item))
            if is_dataclass(item):
                for field in fields(item):
                    visit(getattr(item, field.name))
            elif hasattr(item, "__dict__"):
                visit(vars(item))

    visit(value)
    return tuple(found[key] for key in sorted(found))


def code_identity(value) -> str:
    dependencies = code_dependencies(value)
    if not dependencies:
        raise TypeError(f"{type(value).__qualname__} has no authored ops code identity")
    payload = tuple((item.module, item.symbol, item.fingerprint) for item in dependencies)
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
