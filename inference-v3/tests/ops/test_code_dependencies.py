"""Authored dependency identity contracts; run only at the replacement gate."""

import sys
from dataclasses import dataclass
from types import FunctionType, ModuleType

from ops.compiler.dependencies import code_dependencies, code_identity, invalidate_code_cache


def _helper(value):
    return value + 1


def _consumer(value):
    return _helper(value)


def _unrelated(value):
    return value * 2


def test_line_positions_do_not_change_executable_identity():
    shifted = FunctionType(_consumer.__code__.replace(co_firstlineno=999), _consumer.__globals__)
    shifted.__module__ = _consumer.__module__
    shifted.__qualname__ = _consumer.__qualname__
    assert code_identity(shifted) == code_identity(_consumer)


def test_referenced_helper_changes_identity_without_unrelated_invalidation(monkeypatch):
    invalidate_code_cache()
    before = code_identity(_consumer)
    unrelated = code_identity(_unrelated)

    def changed(value):
        return value + 3

    monkeypatch.setattr(sys.modules[__name__], "_helper", changed)
    invalidate_code_cache()
    assert code_identity(_consumer) != before
    assert code_identity(_unrelated) == unrelated


def test_nested_function_globals_are_dependencies():
    def outer():
        def inner(value):
            return _helper(value)
        return inner

    dependencies = code_dependencies(outer)
    assert any(item.symbol == "_helper" for item in dependencies)


def test_tilelang_style_macro_uses_original_authored_function():
    class Macro:
        orig_func = staticmethod(_helper)

    assert code_dependencies(Macro()) == code_dependencies(_helper)


def test_defaults_are_part_of_loaded_code_identity():
    def function(value=1):
        return value

    replacement = FunctionType(function.__code__, function.__globals__, argdefs=(2,))
    replacement.__module__ = function.__module__
    replacement.__qualname__ = function.__qualname__
    assert code_identity(function) != code_identity(replacement)


def test_callable_body_fields_are_dependencies():
    @dataclass(frozen=True)
    class Body:
        function: object

        def __call__(self, value):
            return self.function(value)

    first = code_dependencies(Body(_helper))
    second = code_dependencies(Body(_unrelated))
    assert first != second
    assert any(item.symbol == "_helper" for item in first)
    assert not any(item.symbol == "_unrelated" for item in first)


def test_generated_dataclass_methods_have_owner_scoped_identity():
    @dataclass(frozen=True)
    class First:
        left: int

    @dataclass(frozen=True)
    class Second:
        right: str

    def body():
        return First(1), Second("two")

    dependencies = code_dependencies(body)
    keys = [(item.module, item.symbol) for item in dependencies]
    assert len(keys) == len(set(keys))
    assert any("First.__repr__.<closure:" in symbol for _, symbol in keys)
    assert any("Second.__repr__.<closure:" in symbol for _, symbol in keys)


def test_function_local_imported_constants_change_identity(monkeypatch):
    module = ModuleType("ops._dependency_constant_fixture")
    module.SCALE = 2
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def direct(value):
        from ops._dependency_constant_fixture import SCALE
        return value * SCALE

    def nested(value):
        import ops._dependency_constant_fixture as settings
        def apply():
            return value * settings.SCALE
        return apply()

    invalidate_code_cache()
    before = code_identity(direct), code_identity(nested)
    module.SCALE = 3
    invalidate_code_cache()
    after = code_identity(direct), code_identity(nested)
    assert all(old != new for old, new in zip(before, after, strict=True))


def test_immutable_bytecode_analysis_survives_definition_refresh(monkeypatch):
    from ops.compiler import dependencies

    dependencies._instructions.cache_clear()
    invalidate_code_cache()
    before = code_identity(_consumer)

    def unexpected_disassembly(*args, **kwargs):
        raise AssertionError("unchanged bytecode was disassembled again")

    monkeypatch.setattr(dependencies.dis, "get_instructions", unexpected_disassembly)
    invalidate_code_cache()
    assert code_identity(_consumer) == before
