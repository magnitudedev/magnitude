"""Explicit inspection of loaded production objects, without changing their execution."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from performance.records import Assembly, Node, digest


def source_files(obj: Any) -> dict[str, str]:
    """Capture the selected module and its static Python dependency closure.

    Reading modules does not import them. Local benchmark edits never enter the
    production fingerprint; source dependencies of production helpers do.
    """
    owner = obj if inspect.isfunction(obj) or inspect.ismethod(obj) else type(obj)
    module = owner.__module__
    roots = [Path(p) for p in sys.path if p]
    result: dict[str, str] = {}
    pending = [module]

    def locate(name):
        for root in roots:
            candidate = root.joinpath(*name.split("."))
            for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                if path.is_file():
                    return path
        return None

    while pending:
        name = pending.pop()
        if name in result:
            continue
        path = locate(name)
        if path is None:
            continue
        text = path.read_text()
        result[name] = text
        package = name.split(".") if path.name == "__init__.py" else name.split(".")[:-1]
        for item in ast.walk(ast.parse(text)):
            imports = []
            if isinstance(item, ast.Import):
                imports = [alias.name for alias in item.names]
            elif isinstance(item, ast.ImportFrom):
                prefix = package[: len(package) - item.level + 1] if item.level else []
                base = ".".join([*prefix, *([item.module] if item.module else [])])
                imports = [base, *(base + "." + alias.name for alias in item.names)]
            pending.extend(
                n
                for n in imports
                if n.split(".")[0]
                in ("magnitude_engine", "mlx_lm", "mlx_vlm", module.split(".")[0])
            )
    if not result:
        raise ValueError(f"no reproducible source for {owner}")
    return result


def portable_code(code):
    from types import CodeType

    return code.replace(
        co_filename="",
        co_consts=tuple(portable_code(c) if isinstance(c, CodeType) else c for c in code.co_consts),
    )


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


def scalar_parameters(obj: Any) -> dict:
    names = (
        "specialize_prefill",
        "partition_tokens",
        "decode_share",
        "decode_tokens",
        "prefill_stall_seconds",
        "max_prefill_ms",
        "max_prefill_tokens",
        "epsilon",
        "eps",
        "bits",
        "group_size",
        "mode",
        "query_heads",
        "kv_heads",
        "head_width",
        "key_heads",
        "value_heads",
        "key_width",
        "value_width",
        "window",
        "heads_per_group",
        "top_k",
        "normalize",
        "heads",
        "width",
        "layers",
        "page_size",
        "slab_pages",
        "max_active",
        "max_queued",
        "prefill_tokens",
        "max_entries",
        "limit",
        "capacity",
        "softcap",
        "embedding_scale",
    )
    return {
        name: value
        for name in names
        if (value := getattr(obj, name, None)) is not None
        and type(value) in (int, float, bool, str)
    }


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


class _Inspector:
    def __init__(self, artifacts: dict):
        self.artifacts = artifacts
        self.nodes: dict[str, Node] = {}
        self.objects: dict[str, Any] = {}
        self.sources: dict[str, str] = {}
        self.seen: dict[tuple[int, str], str] = {}
        self.source_cache = {}
        self.tensors = {}
        # Unidentified weights cannot accidentally pool across captures.
        self.weight_domain = digest(artifacts) if artifacts else uuid4().hex

    def node(self, obj, path, implementation, *, children=None, dependencies=None, parameters=None):
        component = ":".join(implementation.split(":")[:2])
        key = (id(obj), component)
        if key in self.seen and component.startswith(("STATE:", "KV:STORE")):
            return self.seen[key]
        source_owner = obj if inspect.isfunction(obj) or inspect.ismethod(obj) else type(obj)
        module = source_owner.__module__
        if module not in self.source_cache:
            self.source_cache[module] = source_files(obj)
        sources = self.source_cache[module]
        self.sources.update(sources)
        self.seen[key] = path
        self.objects[path] = obj
        self.nodes[path] = Node(
            implementation,
            digest(sources),
            scalar_parameters(obj) | (parameters or {}),
            children or {},
            dependencies or {},
        )
        return path

    def weighted(self, obj, path, implementation, *, children=None, dependencies=None):
        tensor_path = path.split("layers.", 1)[-1] if "layers." in path else path.rsplit(".", 1)[-1]
        excluded = {id(self.objects[p]) for p in (children or {}).values()}
        arrays, visited, operators, extra_sources = {}, {}, {}, {}
        projections = []

        def collect(value, name):
            if value is None or id(value) in visited or id(value) in excluded:
                return
            visited[id(value)] = value
            if hasattr(value, "shape") and hasattr(value, "nbytes"):
                identity = self.tensors.setdefault(id(value), self.weight_domain + ":" + name)
                arrays[name] = {
                    "identity": identity,
                    "shape": list(value.shape),
                    "bytes": value.nbytes,
                    "dtype": str(value.dtype),
                }
                return
            if callable(value):
                owner = (
                    value if inspect.isfunction(value) or inspect.ismethod(value) else type(value)
                )
                operators[name] = {
                    "type": owner.__module__ + "." + owner.__qualname__,
                    "configuration": scalar_parameters(value),
                }
                if inspect.isfunction(value) or inspect.ismethod(value):
                    code = getattr(value, "__code__", None)
                    if code is not None:
                        import marshal

                        extra_sources[name + ":code"] = hashlib.sha256(
                            marshal.dumps(portable_code(code))
                        ).hexdigest()
                module = owner.__module__
                if module not in self.source_cache:
                    self.source_cache[module] = source_files(value)
                extra_sources.update(self.source_cache[module])
            weight = getattr(value, "weight", None)
            if (
                weight is not None
                and len(weight.shape) in (2, 3)
                and (
                    "Linear" in str(type(value).__name__)
                    or type(value).__name__ == "QuantizedProjection"
                    or implementation.split(":")[1].endswith("READOUT")
                )
            ):
                encoding = getattr(value, "encoding", value)
                bits = getattr(encoding, "bits", None)
                width = weight.shape[-1] if bits is None else weight.shape[-1] * 32 // bits
                identity = self.tensors.setdefault(
                    id(weight), self.weight_domain + ":" + name + ".weight"
                )
                projections.append(
                    {
                        "identity": identity,
                        "input_width": width,
                        "output_width": weight.shape[-2],
                        "experts": weight.shape[0] if len(weight.shape) == 3 else None,
                    }
                )
            if inspect.ismethod(value):
                collect(value.__self__, name)
            elif callable(getattr(value, "parameters", None)):
                collect(value.parameters(), name)
                if callable(getattr(value, "children", None)):
                    collect(value.children(), name)
            elif isinstance(value, dict):
                for key, child in sorted(value.items()):
                    collect(child, name + "." + str(key))
            elif isinstance(value, (list, tuple)):
                for i, child in enumerate(value):
                    collect(child, name + "." + str(i))
            elif is_dataclass(value):
                operators.setdefault(
                    name,
                    {
                        "type": type(value).__module__ + "." + type(value).__qualname__,
                        "configuration": {
                            f.name: getattr(value, f.name)
                            for f in fields(value)
                            if type(getattr(value, f.name)) in (int, float, bool, str)
                        },
                    },
                )
                for f in fields(value):
                    collect(getattr(value, f.name), name + "." + f.name)
            elif type(value).__name__ == "LibraryProgram":
                collect(value.call, name + ".call")
            elif inspect.isfunction(value):
                for key, cell in zip(
                    value.__code__.co_freevars, value.__closure__ or (), strict=True
                ):
                    collect(cell.cell_contents, name + "." + key)
            elif type(value).__name__ in ("Qwen35Program", "Gemma4Program"):
                collect(value.norm, name + ".norm")
                collect(value.blocks, name + ".layers")

        collect(obj, tensor_path)
        result = self.node(
            obj,
            path,
            implementation,
            children=children,
            dependencies=dependencies,
            parameters={
                "weights": {"domain": self.weight_domain, "path": tensor_path},
                "arrays": arrays,
                "operators": operators,
                "matrices": projections,
            },
        )
        if extra_sources:
            self.sources.update(extra_sources)
            self.nodes[result] = Node(
                **{
                    **asdict(self.nodes[result]),
                    "source": digest(
                        {"owner": self.nodes[result].source, "dependencies": extra_sources}
                    ),
                }
            )
        return result

    def attention_geometry(self, path, geometry):
        self.nodes[path].parameters.update(geometry)
        for child in self.nodes[path].children.values():
            if self.nodes[child].component == "MODEL:ATTENTION":
                self.attention_geometry(child, geometry)

    @staticmethod
    def element_bytes(obj):
        array = getattr(obj, "scales", getattr(obj, "weight", None))
        if array is None:
            raise ValueError("projection has no numerical representation")
        return array.nbytes // array.size

    @staticmethod
    def output_width(obj):
        return obj.weight.shape[-2]

    def state(self, obj, path):
        kind = type(obj).__name__
        if kind in ("HybridStateStore", "PagedStateStore", "PageStore"):
            pages = obj if kind == "PageStore" else obj.pages
            arena = pages.arena
            kv = self.node(
                arena,
                path + ".kv",
                "KV:STORE:MAG:PAGED",
                parameters={
                    "layers": [asdict(g) for g in arena.layers],
                    "element_bytes": arena.dtype.size,
                    "slab_pages": arena.allocator.slab_pages,
                    "max_pages": arena.allocator.max_pages,
                },
            )
            if not self.nodes[kv].children:
                self.nodes[kv].children.update(
                    {
                        "append": self.node(
                            pages, path + ".append", "KV:APPEND:MAG:CONTIGUOUS_RUNS"
                        ),
                        "branch": self.node(pages, path + ".branch", "KV:BRANCH:MAG:COPY_ON_WRITE"),
                    }
                )
            self.nodes[self.nodes[kv].children["append"]].parameters.update(
                self.nodes[kv].parameters
            )
            if kind in ("PagedStateStore", "PageStore"):
                return kv
            layouts = [
                [{"shape": list(t.shape), "bytes": t.nbytes} for t in layout.tensors]
                for layout in obj.layouts
            ]
            recurrent = self.node(
                obj,
                path + ".recurrent",
                "STATE:RECURRENT:MAG:CHECKPOINTED",
                parameters={"layouts": layouts},
            )
            return self.node(
                obj, path, "STATE:QWEN35:MAG:HYBRID", children={"kv": kv, "recurrent": recurrent}
            )
        if kind == "LibraryStateStore":
            return self.node(
                obj,
                path,
                "STATE:CHECKPOINTS:MAG:NATIVE",
                parameters={
                    "cache_types": [
                        type(c).__module__ + "." + type(c).__name__ for c in obj.make_cache()
                    ],
                },
            )
        raise TypeError(f"no state binding for {kind}")

    def visit(self, obj, path):
        kind = type(obj).__name__
        module = type(obj).__module__
        if kind == "OwnedProgram":
            return self.visit(obj._program, path)
        if kind in ("Qwen35Program", "Gemma4Program"):
            family = "QWEN35" if kind == "Qwen35Program" else "GEMMA4"
            children = {"embedding": self.visit(obj.embedding, path + ".embedding")}
            producers = {}
            if kind == "Gemma4Program" and obj.per_layer is not None:
                children["inputs"] = self.visit(obj.per_layer, path + ".inputs")
            for i, block in enumerate(obj.blocks):
                prefix = f"{path}.layers.{i}"
                mixer = block.mixer if kind == "Qwen35Program" else block.attention
                mp = self.visit(mixer, prefix + ".mixer")
                children[f"layers.{i}.mixer"] = mp
                children[f"layers.{i}.feedforward"] = self.visit(
                    block.feedforward, prefix + ".feedforward"
                )
                if kind == "Gemma4Program":
                    if mixer.producer is not None:
                        producers[mixer.source] = self.nodes[mp].children["producer"]
                    elif mixer.source in producers:
                        self.nodes[mp].dependencies["kv_producer"] = producers[mixer.source]
                    else:
                        raise ValueError("Gemma KV consumer has no earlier producer")
                    producer_path = producers[mixer.source]
                    producer = self.objects[producer_path]
                    width = self.output_width(producer.keys) // producer.heads
                    self.attention_geometry(
                        self.nodes[mp].children["attention"],
                        {
                            "query_heads": mixer.heads,
                            "kv_heads": producer.heads,
                            "key_width": width,
                            "value_width": width,
                            "window": mixer.window,
                            "element_bytes": self.element_bytes(producer.keys),
                            "kv_source": mixer.source,
                        },
                    )
                    if block.layer_input is not None:
                        lp = self.weighted(
                            block.layer_input,
                            prefix + ".inputs",
                            "MODEL:GEMMA4.INPUTS:MAG:PER_LAYER",
                            dependencies={"prepared": children["inputs"]},
                        )
                        children[f"layers.{i}.inputs"] = lp
            children["readout"] = self.weighted(
                obj.output,
                path + ".readout",
                f"MODEL:{family}.READOUT:MAG:"
                + ("STANDARD" if family == "QWEN35" else "SOFTCAPPED"),
            )
            variant = (
                "RESIDENT_COMPILED" if getattr(obj, "decode", None) is not None else "LAYERWISE"
            )
            result = self.weighted(obj, path, f"MODEL:{family}:MAG:{variant}", children=children)
            # Norms and residual scaling belong to the layer assembly, not new nodes.
            self.nodes[result].parameters["layer_count"] = len(obj.blocks)
            return result
        if kind == "GatedAttention":
            geometry = {
                "query_heads": obj.query_heads,
                "kv_heads": obj.kv_heads,
                "key_width": obj.head_width,
                "value_width": obj.head_width,
                "element_bytes": self.element_bytes(obj.queries_and_gate),
            }
            child = self.visit(obj.attention, path + ".attention")
            self.attention_geometry(child, geometry)
            children = {"attention": child}
            return self.weighted(
                obj, path, "MODEL:QWEN35.ATTENTION:MAG:SEPARATE_PROJECTIONS", children=children
            )
        if kind == "RecurrentMixer":
            child = self.visit(obj.operation.graph.recurrence, path + ".update")
            p = scalar_parameters(obj.operation.graph)
            p["element_bytes"] = self.element_bytes(obj.operation.graph.qkv)
            self.nodes[child].parameters.update(p)
            result = self.weighted(
                obj.operation.graph,
                path,
                "MODEL:QWEN35.RECURRENCE:MAG:COMPILED_REGION",
                children={"update": child},
            )
            self.objects[result] = obj
            self.nodes[result].parameters.update(p)
            return result
        if kind == "RoutedFeedForward":
            child = self.visit(obj.experts, path + ".experts")
            self.nodes[child].parameters["top_k"] = obj.top_k
            return self.weighted(
                obj, path, "MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED", children={"experts": child}
            )
        if kind == "DenseFeedForward":
            return self.weighted(obj, path, "MODEL:QWEN35.FEEDFORWARD:LM:DENSE")
        if kind == "GemmaAttention":
            children = {"attention": self.visit(obj.operation, path + ".attention")}
            if obj.producer is not None:
                children["producer"] = self.visit(obj.producer, path + ".producer")
            return self.weighted(
                obj, path, "MODEL:GEMMA4.ATTENTION:MAG:SHARED_KV", children=children
            )
        if kind == "MTPProgram":
            return self.weighted(obj, path, "MODEL:QWEN35.MTP:MAG:CONDITIONED")
        if kind == "GemmaFeedForward":
            children = {"dense": self.visit(obj.dense, path + ".dense")}
            if obj.experts is not None:
                children["experts"] = self.visit(obj.experts, path + ".experts")
            return self.weighted(
                obj, path, "MODEL:GEMMA4.FEEDFORWARD:MAG:BRANCHED", children=children
            )
        if kind == "ExpertBranch":
            child = self.visit(obj.operation, path + ".experts")
            self.nodes[child].parameters["top_k"] = obj.router.top_k
            return self.weighted(
                obj, path, "MODEL:GEMMA4.EXPERT_BRANCH:MAG:ROUTED", children={"experts": child}
            )
        known = {
            "MetalPagedAttention": "MODEL:ATTENTION:MAG:PAGED",
            "GatheredAttention": "MODEL:ATTENTION:MAG:GATHERED",
            "ResidentExperts": "MODEL:EXPERTS:MAG:RESIDENT_GATHERED",
            "ResidentEmbedding": "MODEL:EMBEDDING:MAG:RESIDENT",
            "ResidentAffineEmbedding": "MODEL:EMBEDDING:MAG:RESIDENT",
            "KVProducer": "MODEL:GEMMA4.KV:MAG:PRODUCER",
            "PerLayerInputs": "MODEL:GEMMA4.INPUTS:MAG:PER_LAYER",
            "GeGLU": "MODEL:GEMMA4.MLP:MAG:GEGLU",
            "LibraryProgram": "MODEL:FORWARD:VLM:STANDARD",
        }
        if kind == "LibraryProgram":
            result = self.weighted(obj, path, known[kind])
            self.nodes[result].parameters["opaque"] = True
            return result
        if kind == "MetalPagedAttention":
            fallback = self.visit(obj.prefill, path + ".fallback")
            return self.node(obj, path, known[kind], children={"fallback": fallback})
        if kind in known:
            return (
                self.weighted(obj, path, known[kind])
                if kind not in ("MetalPagedAttention", "GatheredAttention")
                else self.node(obj, path, known[kind])
            )
        if "models.recurrence" in module:
            source = "LM" if kind == "MLXDelta" else "MAG"
            return self.node(
                obj,
                path,
                f"MODEL:GATED_DELTA:{source}:"
                + (
                    "STANDARD"
                    if source == "LM"
                    else "REFERENCE"
                    if kind == "DeltaReference"
                    else "FUSED_UPDATE"
                ),
            )
        raise TypeError(f"no semantic component binding for {module}.{kind}")


def inspect_component(obj, *, path="component", artifacts=None) -> BoundAssembly:
    inspector = _Inspector(artifacts or {})
    root = inspector.visit(obj, path)
    graph = Assembly(
        root, inspector.nodes, inspector.nodes[root].implementation, inspector.artifacts
    )
    return BoundAssembly(graph, inspector.objects, inspector.sources)


def inspect_upstream(model, *, artifact: str) -> BoundAssembly:
    artifacts = {"target": artifact_identity(artifact)}
    inspector = _Inspector(artifacts)
    inspector.weight_domain = digest(artifacts["target"])
    source = "LM" if type(model).__module__.startswith("mlx_lm.") else "VLM"
    root = inspector.weighted(model, "target", f"MODEL:FORWARD:{source}:STANDARD")
    inspector.nodes[root].parameters["opaque"] = True
    return BoundAssembly(
        Assembly(root, inspector.nodes, f"{source} forward", artifacts),
        inspector.objects,
        inspector.sources,
    )


def inspect_engine(residency) -> BoundAssembly:
    properties = residency.properties
    artifacts = {"target": artifact_identity(properties["target_path"])}
    engine = residency.engine
    method = engine.generation.method
    paths = {"target": properties["target_path"]}
    if hasattr(method, "head"):
        for _, candidate in getattr(
            getattr(residency, "resources", None), "_instances", {}
        ).values():
            if getattr(candidate, "program", None) is method.head.program and hasattr(
                candidate, "descriptor"
            ):
                paths["draft"] = candidate.descriptor.path
                artifacts["draft"] = artifact_identity(candidate.descriptor.path)
                break
        if "draft" not in artifacts:
            artifacts["draft"] = {"unbound": uuid4().hex}
    method_identity = engine.generation.method.identity
    for name, path in paths.items():
        method_identity = method_identity.replace(path, digest(artifacts[name]))
    ins = _Inspector(artifacts)
    ins.weight_domain = digest(artifacts["target"])
    target = ins.visit(engine.generation.model.program, "target")
    target_state = ins.state(engine.generation.model.states, "target.state")
    ins.nodes[target].dependencies["state"] = target_state
    scheduler = ins.node(engine.scheduler, "scheduler", "SCHEDULING:SERVICE:MAG:TIME_SHARING")
    prefix = ins.node(engine.prefixes, "prefixes", "CACHE:PREFIX:MAG:CHECKPOINTS")
    memory = ins.node(residency.budget, "memory", "MEMORY:ACCOUNTING:MAG:RESERVATIONS")
    generation = ins.node(
        engine.generation,
        "generation",
        "GENERATION:"
        + (
            "PLAIN:MAG:TARGET"
            if properties["speculative_backend"] in (None, "none")
            else "SPECULATION:MAG:TARGET_MATCHING"
        ),
        children={"target": target},
        parameters={"method": method_identity},
    )
    from magnitude_engine.generation.acceptance import accept_prefix
    from magnitude_engine.generation.execution import serve
    from magnitude_engine.generation.sampling import SequenceSampler

    assembly = ins.node(serve, "batching", "BATCHING:ASSEMBLY:MAG:READY_COMPATIBLE")
    device = ins.node(engine.generation.model.owner, "execution", "EXECUTION:DEVICE:MAG:ASYNC")
    admission = ins.node(engine.submit, "admission", "SCHEDULING:ADMISSION:MAG:FIFO")
    prefill = ins.node(engine.generation.prefill_many, "prefill", "SCHEDULING:PREFILL:MAG:CHUNKED")
    ins.nodes[scheduler].children["prefill"] = prefill
    sampling = ins.node(
        SequenceSampler.__init__, "sampling", "GENERATION:SAMPLING:MAG:POSITION_KEYED"
    )
    ins.nodes[generation].children["sampling"] = sampling
    if properties["speculative_backend"] not in (None, "none"):
        acceptance = ins.node(accept_prefix, "acceptance", "GENERATION:ACCEPTANCE:MAG:PREFIX")
        ins.nodes[generation].children["acceptance"] = acceptance
    method = engine.generation.method
    if hasattr(method, "head"):
        ins.weight_domain = digest(artifacts.get("draft", artifacts["target"]))
        head = ins.visit(method.head.program, "draft")
        head_state = ins.state(method.head.states, "draft.state")
        ins.nodes[head].dependencies["state"] = head_state
        ins.nodes[generation].children["draft"] = head
    ins.nodes[generation].dependencies["execution"] = device
    root = ins.node(
        engine,
        "engine",
        "ENGINE:INFERENCE:MAG:STANDARD",
        children={
            "generation": generation,
            "admission": admission,
            "batching": assembly,
            "execution": device,
            "scheduling": scheduler,
            "prefixes": prefix,
            "memory": memory,
        },
        parameters={
            k: v for k, v in properties.items() if k not in ("target_path", "memory_bytes")
        },
    )
    label = (
        Path(properties["target_path"]).parts[-3]
        if "snapshots" in Path(properties["target_path"]).parts
        else Path(properties["target_path"]).name
    )
    variant = ins.nodes[target].implementation.rsplit(":", 1)[-1]
    graph = Assembly(root, ins.nodes, f"{label} · {variant} · {method_identity}", artifacts)
    return BoundAssembly(graph, ins.objects, ins.sources)


def bind_operation(component, implementation: str, *, parameters=None) -> Binding:
    """Bind a semantic operation where production ownership is a method or function."""
    if isinstance(component, Binding):
        if component.node.component != ":".join(implementation.split(":")[:2]):
            raise ValueError("benchmark and component contracts differ")
        return component
    inspector = _Inspector({})
    path = inspector.node(component, "component", implementation, parameters=parameters)
    graph = Assembly(path, inspector.nodes, implementation)
    return BoundAssembly(graph, inspector.objects, inspector.sources).at(path)


def inspect_state(store, *, path="state") -> BoundAssembly:
    inspector = _Inspector({})
    root = inspector.state(store, path)
    return BoundAssembly(
        Assembly(root, inspector.nodes, inspector.nodes[root].implementation),
        inspector.objects,
        inspector.sources,
    )
