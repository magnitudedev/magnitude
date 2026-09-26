"""Reload authored operation modules without restarting the device owner.

Only original Python source is loaded. No generated kernel source, compiler IR
manipulation or native compilation occurs here. File changes are discovered on
explicit requests; rendering and file saves do not execute device work.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
from threading import RLock
from types import FunctionType

from ..compiler.dependencies import CodeDependency, code_dependencies, invalidate_code_cache


@dataclass(frozen=True, slots=True)
class ModuleSource:
    name: str
    path: Path
    content: bytes
    package: bool

    @property
    def fingerprint(self):
        return hashlib.sha256(self.content).hexdigest()

    @property
    def import_groups(self) -> tuple[tuple[str, ...], ...]:
        """Module candidates for each independent eager import binding."""
        package = self.name if self.package else self.name.rpartition(".")[0]
        groups = []
        def eager_nodes(node):
            # Imports inside function bodies execute on invocation, after modules
            # are installed. They are not module initialization dependencies.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return
            yield node
            for child in ast.iter_child_nodes(node):
                yield from eager_nodes(child)

        for node in eager_nodes(ast.parse(self.content, filename=str(self.path))):
            if isinstance(node, ast.Import):
                groups.extend((alias.name,) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = importlib.util.resolve_name("." * node.level + module, package)
                groups.extend((f"{module}.{alias.name}", module) for alias in node.names)
        return tuple(groups)

    @property
    def imports(self) -> frozenset[str]:
        return frozenset(name for group in self.import_groups for name in group)


class _SnapshotLoader(SourceFileLoader):
    def __init__(self, source: ModuleSource):
        super().__init__(source.name, str(source.path))
        self.source = source

    def get_code(self, fullname):
        # Timestamp pyc caches can execute stale code after same-second,
        # same-size edits. Load the exact captured source without deleting caches.
        return self.source_to_code(self.source.content, str(self.source.path))


@dataclass(frozen=True, slots=True)
class RefreshResult:
    modules: tuple[str, ...]
    changed: frozenset[tuple[str, str]]


class _LoadedSources:
    """Process-owned loaded revision for portable operation definitions.

    Kernel modules and their package are replaceable definitions, not live owners.
    Runtime/fixture/formula objects are retained. Structural changes to those
    owners require a new prepared configuration, not unsafe class hot-patching.
    """

    def __init__(self):
        self._sources: dict[str, ModuleSource] = {}
        self._symbols: dict[tuple[str, str], CodeDependency] = {}

    @staticmethod
    def _read() -> dict[str, ModuleSource]:
        from .. import kernels

        root = Path(kernels.__file__).parent
        sources = {}
        for path in sorted(root.glob("*.py")):
            package = path.name == "__init__.py"
            name = "ops.kernels" if package else f"ops.kernels.{path.stem}"
            sources[name] = ModuleSource(name, path, path.read_bytes(), package)
        source_loop = root.parent / "compiler" / "streaming.py"
        sources["ops.compiler.streaming"] = ModuleSource(
            "ops.compiler.streaming", source_loop, source_loop.read_bytes(), False,
        )
        return sources

    @staticmethod
    def _symbols_for(modules) -> dict[tuple[str, str], CodeDependency]:
        symbols = {}
        for module in modules:
            for value in vars(module).values():
                original = getattr(value, "orig_func", value)
                if isinstance(original, (type, FunctionType)) and original.__module__ == module.__name__:
                    for dependency in code_dependencies(original):
                        symbols[dependency.module, dependency.symbol] = dependency
        return symbols

    def refresh(self, sources: dict[str, ModuleSource]) -> RefreshResult:
        changed_modules = {name for name in self._sources.keys() | sources.keys()
                           if name not in self._sources or name not in sources or
                           self._sources[name].fingerprint != sources[name].fingerprint}
        if not changed_modules:
            return RefreshResult((), frozenset())
        if removed := self._sources.keys() - sources.keys():
            raise ValueError(f"operation modules removed; reopen the configuration: {sorted(removed)}")
        dependencies = {}
        for name, source in sources.items():
            # `from . import child` binds the child module, not the package's
            # re-exported definitions. Resolve each binding independently: another
            # import of a package-level function still depends on the package.
            dependencies[name] = {
                next(module for module in group if module in sources)
                for group in source.import_groups if any(module in sources for module in group)
            }
        # A dependent module's from-import bindings must refer to the new object.
        affected = set(changed_modules)
        while parents := {name for name, imports in dependencies.items()
                          if name not in affected and imports & affected}:
            affected.update(parents)
        ordered = []
        pending = set(affected)
        while pending:
            ready = sorted(name for name in pending if not dependencies[name] & pending)
            if not ready:
                raise ValueError(f"cyclic operation-module imports cannot be refreshed safely: {sorted(pending)}")
            ordered.extend(ready)
            pending.difference_update(ready)
        previous = {name: sys.modules.get(name) for name in ordered}
        parent_attributes = {}
        try:
            for name in ordered:
                source = sources[name]
                loader = _SnapshotLoader(source)
                spec = importlib.util.spec_from_file_location(
                    name, source.path, loader=loader,
                    submodule_search_locations=[str(source.path.parent)] if source.package else None,
                )
                if spec is None:
                    raise ImportError(f"cannot load operation module {name}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                loader.exec_module(module)
                parent_name, _, attribute = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    parent_attributes[name] = parent, attribute, getattr(parent, attribute, None)
                    setattr(parent, attribute, module)
            invalidate_code_cache()
            symbols = self._symbols_for(sys.modules[name] for name in sources)
            changed = frozenset(key for key in self._symbols.keys() | symbols.keys()
                                if self._symbols.get(key) != symbols.get(key))
            self._sources, self._symbols = dict(sources), symbols
            return RefreshResult(tuple(ordered), changed)
        except BaseException:
            for name in reversed(ordered):
                old = previous[name]
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old
                if name in parent_attributes:
                    parent, attribute, value = parent_attributes[name]
                    if value is None:
                        delattr(parent, attribute)
                    else:
                        setattr(parent, attribute, value)
            invalidate_code_cache()
            raise


_LOADED: dict[tuple[tuple[str, str], ...], _LoadedSources] = {}
_SOURCE_LOCK = RLock()


class OperationSources:
    """Each worker observes revisions of the same process-wide Python modules.

    Reload installation is shared, but changed-symbol delivery is per worker.
    Thus a second worker sees an edit already installed by the first, including
    edits reverted before its next request. Live runtimes are never reloaded.
    """

    _read = staticmethod(_LoadedSources._read)

    def __init__(self):
        self._symbols: dict[tuple[str, str], CodeDependency] = {}
        self._observed_sources: dict[str, ModuleSource] = {}

    def pending_modules(self) -> frozenset[str]:
        """Read-only edit notice; it does not import, build, compile or measure."""
        if not self._observed_sources:
            return frozenset()
        current = self._read()
        return frozenset(name for name in self._observed_sources.keys() | current.keys()
                         if name not in current or name not in self._observed_sources or
                         current[name].fingerprint != self._observed_sources[name].fingerprint)

    def refresh(self) -> RefreshResult:
        with _SOURCE_LOCK:
            sources = self._read()
            # Group by source roots, not current filenames: deleting a module
            # must reach the existing owner's removal check.
            roots = tuple(sorted({(name.split(".")[0], str(source.path.parent))
                                  for name, source in sources.items()}))
            loaded = _LOADED.setdefault(roots, _LoadedSources())
            installed = loaded.refresh(sources)
            symbols = loaded._symbols
            changed = frozenset(key for key in self._symbols.keys() | symbols.keys()
                                if self._symbols.get(key) != symbols.get(key))
            self._symbols = dict(symbols)
            self._observed_sources = dict(sources)
            return RefreshResult(installed.modules, changed)
