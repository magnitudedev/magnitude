"""Fresh source loading, dependent bindings and failed-refresh rollback."""

import sys
from pathlib import Path

import pytest

from ops.lab.refresh import ModuleSource, OperationSources


def test_source_refresh_rebinds_imports_and_rolls_back(monkeypatch, tmp_path):
    package = "ops_refresh_fixture"
    names = (package, package + ".helper", package + ".consumer")
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    sources = {
        package: ModuleSource(package, tmp_path / "__init__.py", b"", True),
        names[1]: ModuleSource(names[1], tmp_path / "helper.py", b"def value():\n    return 1\n", False),
        names[2]: ModuleSource(names[2], tmp_path / "consumer.py",
                               b"from .helper import value\ndef invoke():\n    return value()\n", False),
    }
    refresh = OperationSources()
    monkeypatch.setattr(refresh, "_read", lambda: sources)
    try:
        first = refresh.refresh()
        assert first.modules
        assert sys.modules[names[2]].invoke() == 1
        observer = OperationSources()
        monkeypatch.setattr(observer, "_read", lambda: sources)
        assert observer.refresh().changed
        sources[names[1]] = ModuleSource(names[1], tmp_path / "helper.py",
                                         b"def value():\n    return 2\n", False)
        second = refresh.refresh()
        assert names[2] in second.modules
        assert sys.modules[names[2]].invoke() == 2
        observed = observer.refresh()
        assert observed.modules == ()
        assert (names[1], "value") in observed.changed
        assert refresh.refresh().modules == ()
        successful = sys.modules[names[2]]
        sources[names[1]] = ModuleSource(names[1], tmp_path / "helper.py",
                                         b"raise RuntimeError('bad source revision')\n", False)
        with pytest.raises(RuntimeError, match="bad source"):
            refresh.refresh()
        assert sys.modules[names[2]] is successful
        assert sys.modules[names[2]].invoke() == 2
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_source_import_graph_is_read_without_executing_source():
    source = ModuleSource("ops.example.consumer", Path("consumer.py"),
                          b"from .helper import value\nraise RuntimeError('not executed')\n", False)
    assert "ops.example.helper" in source.imports
