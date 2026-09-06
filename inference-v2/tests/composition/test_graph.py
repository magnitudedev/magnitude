import json
from dataclasses import FrozenInstanceError, replace

import pytest

from magnitude_engine.composition import Blueprint, Catalog, build, component, digest, dumps, loads


class Value:
    events = []

    def __init__(self, *, number: int):
        self.number = number
        self.events.append(("open", number))

    def close(self):
        self.events.append(("close", self.number))


class Pair:
    def __init__(self, *, left: Value, right: Value, fail: bool):
        if fail:
            raise RuntimeError("construction failed")
        self.left, self.right = left, right


@component
class Leaf(Blueprint[Value]):
    number: int

    @staticmethod
    def implementation():
        return Value


@component
class Branch(Blueprint[Pair]):
    left: Blueprint[Value]
    right: Blueprint[Value]
    fail: bool = False

    @staticmethod
    def implementation():
        return Pair


CATALOG = Catalog((Leaf, Branch))


def test_sharing_survives_validated_roundtrip_and_construction():
    Value.events.clear()
    leaf = Leaf(number=3)
    root = Branch(left=leaf, right=leaf)
    restored = loads(dumps(root), CATALOG)
    assert restored.left is restored.right
    assert digest(root) == digest(restored)
    with build(restored) as pair:
        assert pair.left is pair.right
        assert pair.left.number == 3
    assert Value.events == [("open", 3), ("close", 3)]
    with build(restored) as second:
        assert second.left is not pair.left


def test_equal_independent_nodes_remain_independent():
    root = Branch(left=Leaf(number=1), right=Leaf(number=1))
    restored = loads(dumps(root), CATALOG)
    assert restored.left is not restored.right
    with build(restored) as pair:
        assert pair.left is not pair.right


def test_frozen_strict_fields_and_dependency_contracts():
    leaf = Leaf(number=2)
    with pytest.raises(FrozenInstanceError):
        leaf.number = 3
    assert replace(leaf, number=4).number == 4
    with pytest.raises(ValueError):
        Leaf(number="2")
    with pytest.raises(TypeError):
        Branch(left=Branch(left=leaf, right=leaf), right=leaf)


def test_failed_parent_retires_shared_dependencies_once():
    Value.events.clear()
    leaf = Leaf(number=9)
    with (
        pytest.raises(RuntimeError, match="construction failed"),
        build(Branch(left=leaf, right=leaf, fail=True)),
    ):
        pytest.fail("construction must fail before publishing")
    assert Value.events == [("open", 9), ("close", 9)]


@pytest.mark.parametrize("corruption", ["cycle", "dangling", "unknown", "duplicate", "extra"])
def test_invalid_graph_never_constructs_runtime(corruption):
    Value.events.clear()
    leaf = Leaf(number=1)
    graph = json.loads(dumps(Branch(left=leaf, right=leaf)))
    if corruption == "cycle":
        graph["nodes"][0]["fields"]["left"] = {"ref": "n0"}
    elif corruption == "dangling":
        graph["nodes"][0]["fields"]["left"] = {"ref": "absent"}
    elif corruption == "unknown":
        graph["nodes"][0]["type"] = "os.system"
    elif corruption == "duplicate":
        graph["nodes"].append(graph["nodes"][1])
    else:
        extra = dict(graph["nodes"][1], id="unused")
        graph["nodes"].append(extra)
    with pytest.raises(ValueError):
        loads(json.dumps(graph), CATALOG)
    assert Value.events == []


def test_all_wiring_is_checked_before_any_runtime_construction():
    @component
    class Broken(Blueprint[Value]):
        wrong: int

        @staticmethod
        def implementation():
            return Value

    Value.events.clear()
    with pytest.raises(TypeError), build(Branch(left=Leaf(number=1), right=Broken(wrong=2))):
        pytest.fail("bad constructor wiring was accepted")
    assert Value.events == []


def test_cleanup_continues_and_preserves_primary_and_cleanup_failures():
    events = []

    class Resource:
        def __init__(self, *, number: int):
            self.number = number

        def close(self):
            events.append(self.number)
            raise ValueError(f"close {self.number}")

    @component
    class ResourceBP(Blueprint[Resource]):
        number: int

        @staticmethod
        def implementation():
            return Resource

    class Parent:
        def __init__(self, *, children: tuple[Resource, ...]):
            raise RuntimeError("parent failed")

    @component
    class ParentBP(Blueprint[Parent]):
        children: tuple[Blueprint[Resource], ...]

        @staticmethod
        def implementation():
            return Parent

    with (
        pytest.raises(ExceptionGroup) as error,
        build(
            ParentBP(children=(ResourceBP(number=1), ResourceBP(number=2))),
        ),
    ):
        pytest.fail("parent constructor should fail")
    assert events == [2, 1]
    assert [str(e) for e in error.value.exceptions] == ["parent failed", "close 2", "close 1"]
