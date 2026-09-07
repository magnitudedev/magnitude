"""Capture declared production components; serialize facts, never reconstruct a model."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from magnitude_engine.components import component_id, component_of
from magnitude_engine.models.architectures.mlx_vlm.program import LibraryProgram
from performance.bindings import Fields, Use, foreign, read, resolve
from performance.facts import NeuralParameters, OpaqueParameters, TensorFacts
from performance.parameters import materialize, tensors
from performance.records import Assembly, CompositionOrigin, Node, digest


def source_files(obj: Any) -> dict[str, str]:
    """Source archives are provenance; source_key selects executable symbols separately."""
    owner = (
        obj
        if inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isclass(obj)
        else type(obj)
    )
    path = inspect.getsourcefile(owner)
    return {owner.__module__: Path(path).read_text()} if path else {}


def source_key(owners: tuple[object, ...]) -> tuple[str, dict[str, str]]:
    symbols, files, seen = {}, {}, set()

    class RuntimeCode(ast.NodeTransformer):
        def visit_ClassDef(self, node):
            node.decorator_list = [
                d
                for d in node.decorator_list
                if not (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id == "component"
                )
            ]
            return self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
            node.decorator_list = [
                d
                for d in node.decorator_list
                if not (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id == "component"
                )
            ]
            if node.name == "bindings":
                return None
            node.returns = None
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                arg.annotation = None
            if node.args.vararg:
                node.args.vararg.annotation = None
            if node.args.kwarg:
                node.args.kwarg.annotation = None
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            return self.visit_FunctionDef(node)

        def visit_AnnAssign(self, node):
            return ast.Assign(targets=[node.target], value=node.value) if node.value else None

        def visit_Expr(self, node):
            return (
                None
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                else self.generic_visit(node)
            )

    def visit(value):
        value = inspect.unwrap(value)
        owner = (
            value
            if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value)
            else type(value)
        )
        if id(owner) in seen:
            return
        seen.add(id(owner))
        module = owner.__module__
        if not module.startswith(
            ("magnitude_engine", "mlx_lm", "mlx_vlm", "mlx.nn", "performance", "tests")
        ):
            return
        try:
            tree = RuntimeCode().visit(ast.parse(textwrap.dedent(inspect.getsource(owner))))
        except (OSError, TypeError):
            return
        key = module + "." + owner.__qualname__
        symbols[key] = ast.dump(tree, include_attributes=False)
        files.update(source_files(owner))
        namespace = vars(inspect.getmodule(owner))
        # Follow referenced global executable symbols, not every import in the file.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in namespace:
                dependency = namespace[node.id]
                if (
                    callable(dependency) and inspect.isfunction(inspect.unwrap(dependency))
                ) or inspect.isclass(dependency):
                    visit(dependency)
                elif isinstance(dependency, (str, int, float, bool, tuple)):
                    try:
                        symbols[module + "." + node.id] = json.dumps(dependency, allow_nan=False)
                    except (TypeError, ValueError):
                        pass
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                parent = namespace.get(node.value.id)
                if inspect.ismodule(parent):
                    dependency = getattr(parent, node.attr, None)
                    if (
                        callable(dependency) and inspect.isfunction(inspect.unwrap(dependency))
                    ) or inspect.isclass(dependency):
                        visit(dependency)

    for owner in owners:
        visit(owner)
    if not symbols:
        raise ValueError("component has no reproducible executable source")
    return digest(symbols), files


def artifact_identity(directory: str) -> dict:
    """Use immutable hub revisions; local artifacts require content verification."""
    path = Path(directory).expanduser().resolve()
    if path.parent.name == "snapshots":
        revision = path.name
    else:
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise ValueError(f"no model weights at {path}")
        signature = [
            (f.name, f.stat().st_size, f.stat().st_mtime_ns, f.stat().st_ctime_ns, f.stat().st_ino)
            for f in files
        ]
        cache = (
            Path.home() / ".cache/magnitude/performance/artifacts" / (digest(str(path)) + ".json")
        )
        previous = json.loads(cache.read_text()) if cache.exists() else {}
        if previous.get("signature") == json.loads(json.dumps(signature)):
            revision = previous["revision"]
        else:
            content = {}
            for file in files:
                h = hashlib.sha256()
                with file.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 << 20), b""):
                        h.update(block)
                content[file.name] = h.hexdigest()
            revision = digest(content)
            from performance.store import atomic

            atomic(cache, {"signature": signature, "revision": revision})
    config = json.loads((path / "config.json").read_text())
    return {"revision": revision, "config": config}


@dataclass
class BoundAssembly:
    graph: Assembly
    objects: dict[str, Any]
    sources: dict[str, str]

    def at(self, path: str) -> Binding:
        if path not in self.graph.nodes:
            raise KeyError(path)
        return Binding(self, path)


@dataclass(frozen=True)
class Binding:
    assembly: BoundAssembly
    path: str

    @property
    def node(self) -> Node:
        return self.assembly.graph.nodes[self.path]

    @property
    def instance(self) -> Any:
        return self.assembly.objects[self.path]


class _Capture:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.nodes, self.objects, self.sources = {}, {}, {}
        self.seen, self.tensors, self.code = {}, {}, {}
        self.uses: list[Use] = []
        self.domain = digest(artifacts) if artifacts else uuid4().hex

    def visit(self, reference: Use, path: str) -> str:
        reference, fields, implementation = read(reference)
        key = (
            id(reference.value),
            id(reference.context),
            str(implementation),
            tuple((k, id(v.value)) for k, v in reference.dependencies.items()),
        )
        if key in self.seen:
            return self.seen[key]
        self.seen[key] = path
        # Keep contexts alive: repeated shared kernels may use distinct layer geometry.
        self.uses.append(reference)
        children = {
            role: self.visit(child, path + "." + role) for role, child in fields.children.items()
        }
        dependencies = {
            role: self.visit(child, path + "." + role)
            for role, child in {**fields.dependencies, **reference.dependencies}.items()
        }
        parameters = fields.parameters
        live_tensors = {}
        if isinstance(parameters, NeuralParameters):
            parameters, live_tensors = materialize(parameters, fields.operands)
        elif isinstance(parameters, OpaqueParameters):
            live_tensors = {
                f"{role}.{name}": tensor
                for role, operand in fields.operands.items()
                for name, tensor in tensors(operand).items()
            }
            parameters = parameters.model_copy(
                update={
                    "arrays": {
                        name: TensorFacts(
                            identity=name, shape=t.shape, bytes=t.nbytes, dtype=str(t.dtype)
                        )
                        for name, t in live_tensors.items()
                    }
                }
            )
        if isinstance(parameters, (NeuralParameters, OpaqueParameters)):
            arrays = {}
            for name, fact in parameters.arrays.items():
                tensor = live_tensors.get(name)
                relative = path.removeprefix("engine.generation.")
                role, _, suffix = relative.partition(".")
                artifact = self.artifacts.get(role, self.artifacts.get("target"))
                domain = digest(artifact) if artifact else self.domain
                identity = domain + ":" + suffix + ":" + name
                if tensor is not None:
                    identity = self.tensors.setdefault(id(tensor), identity)
                arrays[name] = fact.model_copy(update={"identity": identity})
            parameters = parameters.model_copy(update={"arrays": arrays})
        owners = (
            reference.value,
            *((reference.declaration,) if reference.declaration is not None else ()),
            *fields.sources,
            *(
                o.__self__ if inspect.ismethod(o) else o
                for o in fields.operands.values()
                if o is not None
            ),
        )
        code_key = tuple(
            id(o if inspect.isfunction(o) or inspect.ismethod(o) or inspect.isclass(o) else type(o))
            for o in owners
        )
        if code_key not in self.code:
            self.code[code_key] = source_key(owners)
        source, files = self.code[code_key]
        self.sources.update(files)
        self.nodes[path] = Node(
            implementation,
            source,
            parameters,
            children,
            dependencies,
            configuration=fields.configuration,
        )
        self.objects[path] = reference.value
        return path


def inspect_component(
    obj: object, *, context=None, path="component", artifacts=None
) -> BoundAssembly:
    from performance import schemas  # noqa: F401 - explicitly install analysis schemas

    capture = _Capture(artifacts or {})
    reference = obj if isinstance(obj, Use) else Use(obj, context)
    root = capture.visit(reference, path)
    declaration = None
    try:
        declaration = component_of(resolve(reference).value)
    except TypeError:
        pass  # Upstream operation adapters have no production model definition.
    definition = declaration.model if declaration is not None else None
    origin = CompositionOrigin(definition.identity, "model") if definition is not None else None
    return BoundAssembly(
        Assembly(
            root, capture.nodes, capture.nodes[root].implementation, capture.artifacts, origin
        ),
        capture.objects,
        capture.sources,
    )


def inspect_upstream(model, *, artifact: str) -> BoundAssembly:
    from magnitude_engine.models.architectures.mlx_vlm.definition import DEFINITION

    artifacts = {"target": artifact_identity(artifact)}
    from performance.benchmarks.references import LMForward

    declaration = LMForward if type(model).__module__.startswith("mlx_lm.") else LibraryProgram
    source = component_id(declaration).rsplit(":", 2)[1]
    bound = inspect_component(
        foreign(
            model,
            declaration,
            Fields(OpaqueParameters(), operands={"model": model}),
        ),
        path="target",
        artifacts=artifacts,
    )
    from dataclasses import replace

    bound.graph = replace(
        bound.graph,
        label=f"{source} forward",
        origin=CompositionOrigin(DEFINITION.identity, "upstream"),
    )
    return bound


def inspect_engine(residency) -> BoundAssembly:
    from performance import schemas  # noqa: F401

    artifacts = {"target": artifact_identity(residency.properties["target_path"])}
    from magnitude_engine.generation.methods.mtp.runtime import MTPMethod

    method = residency.engine.generation.method
    if isinstance(method, MTPMethod):
        artifacts["draft"] = artifact_identity(method.artifact_path)
    capture = _Capture(artifacts)
    root = capture.visit(Use(residency.engine, residency), "engine")
    target = resolve(Use(residency.engine.generation.model.program)).value
    definition = component_of(target).model
    if definition is None:
        raise TypeError("engine target must declare its production model definition")
    # Preserve public occurrence paths without encoding another architecture.
    renames = {
        p: p.removeprefix("engine.generation.")
        if p.startswith(("engine.generation.target", "engine.generation.draft"))
        else p.removeprefix("engine.")
        for p in capture.nodes
    }
    renames["engine"] = "engine"
    from dataclasses import replace

    nodes = {
        renames[p]: replace(
            n,
            children={k: renames[v] for k, v in n.children.items()},
            dependencies={k: renames[v] for k, v in n.dependencies.items()},
        )
        for p, n in capture.nodes.items()
    }
    objects = {renames[p]: v for p, v in capture.objects.items()}
    path = Path(residency.properties["target_path"])
    label = path.parts[-3].split("--")[-1] if "snapshots" in path.parts else path.name
    origin = CompositionOrigin(
        definition.identity, "engine", "default", getattr(residency, "selection", "candidate")
    )
    graph = Assembly(root, nodes, label, artifacts, origin)
    return BoundAssembly(graph, objects, capture.sources)


def bind_operation(component, declaration: object, *, parameters=None) -> Binding:
    expected = component_id(declaration)
    if isinstance(component, Binding):
        if component.node.component != expected.kind:
            raise ValueError("benchmark and component contracts differ")
        return component
    from performance import schemas  # noqa: F401
    from performance.bindings import SCHEMAS, operation

    try:
        selected = component_of(component)
    except TypeError:
        selected = None
    if selected is not None:
        if selected.id.kind != expected.kind:
            raise ValueError("benchmark and component contracts differ")
        shape = SCHEMAS.get(type(component))
        if shape is not None:
            context = None if isinstance(None, shape.context) else parameters
            return inspect_component(component, context=context).at("component")
        # Declared function ports use their actual identity, not a benchmark's default variant.
        return inspect_component(operation(component, parameters)).at("component")
    # External controls cannot carry our declaration; their adapter is explicit.
    if parameters is None:
        from performance.theory.catalog import parameter_type

        parameters = parameter_type(expected.kind)()
    return inspect_component(foreign(component, declaration, Fields(parameters))).at("component")


def inspect_state(store, *, path="state") -> BoundAssembly:
    return inspect_component(store, path=path)
