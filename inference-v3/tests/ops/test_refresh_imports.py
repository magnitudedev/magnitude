"""Reload ordering follows imports evaluated during module initialization."""

import sys
from types import ModuleType

from ops.lab.refresh import ModuleSource, _LoadedSources


def test_function_local_import_does_not_create_false_module_cycle(tmp_path):
    package = 'roofline_refresh_test'
    bodies = {
        package: 'from . import body\n',
        package + '.body': 'from . import schedules\ndef value():\n    return schedules.size()\n',
        package + '.schedules': 'def size():\n    from . import body\n    return 64\n',
    }
    sources = {}
    for name, body in bodies.items():
        path = tmp_path / ('__init__.py' if name == package else name.rsplit('.', 1)[1] + '.py')
        path.write_text(body)
        sources[name] = ModuleSource(name, path, body.encode(), name == package)
    initial = ModuleType(package)
    initial.__path__ = [str(tmp_path)]
    sys.modules[package] = initial
    loaded = _LoadedSources()
    try:
        loaded.refresh(sources)
        assert sys.modules[package].body.value() == 64
        name = package + '.schedules'
        before = sources[name]
        sources[name] = ModuleSource(name, before.path, before.content.replace(b'64', b'128'), False)
        loaded.refresh(sources)
        assert sys.modules[package].body.value() == 128
    finally:
        for name in sources:
            sys.modules.pop(name, None)
