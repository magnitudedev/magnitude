"""Explicit device probes measured and persisted by the ordinary formula runner.

Achieved rates are empirical evidence, not vendor physical peaks or a performance
predictor. Cache conditions and working sets travel with every rate.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from threading import Event
from typing import Literal
from uuid import uuid4

import numpy as np
import tilelang.language as T
from pydantic import Field, model_validator

from ..compiler.compilation import CompileOptions
from ..compiler.dependencies import code_dependencies, code_identity
from ..binding import Binding, DenseImport, Residency, SourceInfo, SourcePlane, SourceSpan
from ..formula import FormulaTree, Unit, formula, quantity, units
from ..operation import operation
from ..tensor import ops as tensor
from ..tensor.tracing import Argument, Signature, trace
from ..tensor.graph import ValueKind
from ..tensor.types import DType, TensorSpec
from ..performance.resources import Resource
from .fixtures import Fixture
from .ownership import exclusive_measurement
from .records import MeasurementProtocol, Outcome, Record, UnavailableMetric


class ProbeProtocol(Record):
    version: Literal[1, 2, 3, 4, 5] = 5
    copy_sizes: tuple[int, ...] = (64 << 10, 128 << 20)
    matrix_width: int = Field(default=2048, ge=32)
    vector_width: int = Field(default=65536, ge=256)
    measurement: MeasurementProtocol = MeasurementProtocol(absolute_tolerance=2e-5, relative_tolerance=2e-5,
                                                          minimum_warmup_seconds=1)

    @classmethod
    def for_capacity(cls, available_bytes):
        """Choose explicit, recorded probe geometry within the live allocation plan."""
        if available_bytes < 1 << 20:
            raise ValueError("resource characterization requires at least 1 MiB of available capacity")
        largest = min(128 << 20, 1 << ((available_bytes // 8).bit_length() - 1))
        width = min(2048, 1 << (math.isqrt(available_bytes // 64).bit_length() - 1))
        return cls(copy_sizes=tuple(dict.fromkeys((64 << 10, largest))), matrix_width=width,
                   vector_width=min(65536, available_bytes // 64))

    @model_validator(mode="after")
    def bounded_working_sets(self):
        if (not 1 <= len(self.copy_sizes) <= 4 or len(set(self.copy_sizes)) != len(self.copy_sizes)
                or any(type(size) is not int or size < 4096 for size in self.copy_sizes)):
            raise ValueError("copy characterization requires one to four distinct working sets of at least 4 KiB")
        return self


class Rate(Record):
    resource: Resource
    value: float = Field(gt=0)
    unit: Unit
    measurement: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    working_set_bytes: int = Field(gt=0)
    dtype: DType
    conditions: tuple[str, ...]
    source: SourceInfo | None = None


class Characterization(Record):
    schema_version: Literal[1] = 1
    identity: str = Field(min_length=1)
    key: str = Field(min_length=1)
    created: datetime
    device: str = Field(min_length=1)
    compiler: str = Field(min_length=1)
    capabilities: str = Field(min_length=1)
    protocol: ProbeProtocol
    rates: tuple[Rate, ...]
    unavailable: tuple[UnavailableMetric, ...] = ()

    @model_validator(mode="after")
    def valid_characterization(self):
        if self.created.utcoffset() is None:
            raise ValueError("characterization timestamp must identify its timezone")
        if not self.rates:
            raise ValueError("characterization requires actual successful probe evidence")
        if len({(rate.resource, rate.dtype, rate.working_set_bytes, rate.source) for rate in self.rates}) != len(self.rates):
            raise ValueError("each probe resource/precision/working-set is characterized once")
        return self


@formula(id="device-probe.copy", version=1)
def copy(source, destination, extent):
    quantity("copy-bytes", 2 * source.shape[0], unit=units.byte)
    return tensor.byte_copy(source, destination, extent)


@formula(id="device-probe.memory", version=1)
def memory(source):
    return tensor.add(source, source)


@formula(id="device-probe.matrix", version=2)
def matrix(left, right):
    result = tensor.linear(left, right, output_dtype=DType.F32)
    for _ in range(63):
        result = tensor.add(result, tensor.linear(left, right, output_dtype=DType.F32))
    return result


def matrix_probe(context):
    """Keep native operand fragments live across the arithmetic chain."""
    left, right = context.inputs
    m, k = left.spec.shape
    n = right.spec.shape[0]
    dtype = left.spec.dtype.value
    instruction = next(item for item in context.lowering.capabilities.matrix_instructions
                       if item.input_dtype == left.spec.dtype and item.accumulation_dtype == DType.F32)
    # Enough independent accumulator chains to expose arithmetic throughput,
    # rather than the latency of repeatedly updating four matrix fragments.
    bm, bn = instruction.m * 8, instruction.n * 8
    threads = context.lowering.capabilities.subgroup_width * 4
    repetitions = sum(context.graph.node(node).operation == "linear" for node in context.nodes)

    @T.macro
    def body(source, weight, output):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            a = T.alloc_fragment((bm, k), dtype)
            b = T.alloc_fragment((bn, k), dtype)
            accum = T.alloc_fragment((bm, bn), "float32")
            for i, reduction in T.Parallel(bm, k):
                a[i, reduction] = T.if_then_else(by * bm + i < m, source[by * bm + i, reduction], 0)
            for j, reduction in T.Parallel(bn, k):
                b[j, reduction] = T.if_then_else(bx * bn + j < n, weight[bx * bn + j, reduction], 0)
            T.clear(accum)
            for _ in T.serial(repetitions):
                T.gemm(a, b, accum, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < m and bx * bn + j < n:
                    output[by * bm + i, bx * bn + j] = accum[i, j]

    return context.kernel(body)


@formula(id="device-probe.arithmetic", version=1)
def arithmetic(left, right):
    result = left
    for _ in range(32):
        result = tensor.add(tensor.multiply(result, right), left)
    return result


@formula(id="device-probe.special", version=1)
def special(value):
    result = value
    for _ in range(16):
        result = tensor.sigmoid(result)
    return result


@formula(id="device-probe.comparisons", version=1)
def comparisons(left, right):
    result = left
    for _ in range(32):
        result = tensor.add(tensor.cast(tensor.less(result, right), DType.F32), left)
    return result


def pointwise_probe(context):
    from ..kernels.fusion import pointwise

    return pointwise(context)


operation(arithmetic)(pointwise_probe)
operation(special)(pointwise_probe)
operation(comparisons)(pointwise_probe)
operation(memory)(pointwise_probe)
operation(matrix)(matrix_probe)


def characterize(device, store, *, protocol: ProbeProtocol = ProbeProtocol(),
                 refresh: bool = False, cancellation: Event | None = None) -> Characterization:
    """Run only on the device owner, explicitly, never inside ordinary execution."""
    from .runner import MeasurementCancelled, MeasurementRunner

    device.check()
    if cancellation is not None and cancellation.is_set():
        raise MeasurementCancelled("device characterization cancelled before preparation")
    from .refresh import OperationSources

    # Install edits before reading the probe-body identities. Probe refresh is
    # explicit and uses the same process-wide source ownership as ordinary Lab.
    with exclusive_measurement():
        OperationSources().refresh()
    identity = device.evidence_identity
    payload = (identity, device.compiler_identity, device.capabilities.fingerprint,
               protocol.model_dump(mode="json"), code_identity(copy.function),
               code_identity(matrix.function), code_identity(arithmetic.function),
               code_identity(special.function), code_identity(comparisons.function), code_identity(memory.function))
    # Ordinary operation edits do not move the denominator. Each probe's actual
    # implementation is retained in its measurement; refresh is deliberate.
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if not refresh:
        cached = store.characterization(key)
        if cached is not None:
            return cached
    rates, unavailable = [], []

    def probe(function, specs, arrays, resource, metric, dtype, *, bindings=None, body=None):
        if cancellation is not None and cancellation.is_set():
            raise MeasurementCancelled("device characterization cancelled")
        graph = trace(function, Signature(tuple(specs)))
        # Arrays follow the declared positional order, which may mix resources
        # and ordinary inputs. Resolve those names rather than category order.
        values = {next(value.id for value in graph.values if value.producer is None and value.name == argument.name): array
                  for argument, array in zip(specs, arrays, strict=True)}
        physical = {next(value.id for value in graph.values if value.producer is None and value.name == name): binding
                    for name, binding in (bindings or {}).items()}
        fixture = Fixture(graph, values, bindings=physical)
        runner = MeasurementRunner(fixture, device, store, CompileOptions(mode="prefill"),
                                   protocol=protocol.measurement, prepared_limit=1, _resource_probe=True)
        try:
            runner.refresh()
            target, = FormulaTree(graph).roots
            # A cancelled/failed characterization may already have persisted
            # checked probes. Their series fixes the math, inputs, device and
            # protocol. Reuse those resource references instead of rerunning
            # successful GPU work; explicit refresh requests new evidence.
            history = store.history(runner._series(fixture.boundary(target))) if not refresh else None
            measurement = history.latest_success if history is not None else None
            if measurement is not None and body is not None:
                prior = {(item.module, item.symbol): item.fingerprint for item in measurement.implementation.authored}
                if any(prior.get((item.module, item.symbol)) != item.fingerprint for item in code_dependencies(body)):
                    measurement = None
            if measurement is None or measurement.implementation.compiler != device.compiler_identity:
                measurement = runner.measure(target, cancellation=cancellation).measurement
            if measurement.outcome != Outcome.COMPLETE:
                raise RuntimeError(measurement.error or "device probe did not complete")
            available = {item.name: item for item in measurement.metrics}
            native_metric = f"kernel-{metric}"
            native = resource != Resource.SOURCE_IMPORT and native_metric in available
            observed = available[native_metric if native else metric]
            rates.append(Rate(resource=resource, value=observed.value, unit=observed.unit,
                              measurement=measurement.identity, metric=observed.name, dtype=dtype,
                              working_set_bytes=(sum(argument.spec.storage_nbytes for argument in specs) +
                                                 sum(graph.value(value).spec.storage_nbytes for value in set(graph.outputs)
                                                     if graph.value(value).resource_id is None)),
                              conditions=(f"{'source-backed input with recurring import' if physical else 'resident inputs'}; at least {protocol.measurement.warmups} warmups over {protocol.measurement.minimum_warmup_seconds:g}s then {protocol.measurement.samples} complete invocations",
                                          ("native compute-pass duration excludes host dispatch and completion gaps" if native else
                                           "host wall duration includes dispatch, completion and output cleanup"),
                                          "cache residency uncontrolled; not a proven DRAM or physical arithmetic peak")))
        finally:
            runner.close()

    with exclusive_measurement():
        for size in protocol.copy_sizes:
            if size * 6 + 32 > device.available_bytes:
                raise ValueError("device probe working set exceeds available capacity; choose an explicit smaller protocol")
            spec = TensorSpec((size // 4,), DType.F32)
            probe(memory, (Argument(spec, "source"),), (np.ones(spec.shape, dtype=np.float32),),
                  Resource.EXECUTION_COPY, "rate:boundary-bytes", DType.F32)
        supported = {instruction.input_dtype for instruction in device.capabilities.matrix_instructions
                     if instruction.accumulation_dtype == DType.F32}
        for dtype in (DType.F16, DType.BF16, DType.F32):
            if dtype not in supported:
                unavailable.append(UnavailableMetric(name=f"matrix:{dtype.value}",
                                                      reason="no declared portable matrix instruction for this precision"))
                continue
            width = protocol.matrix_width
            instruction = next(item for item in device.capabilities.matrix_instructions
                               if item.input_dtype == dtype and item.accumulation_dtype == DType.F32)
            spec = TensorSpec((width, instruction.k * 4), dtype)
            if 4 * (2 * spec.storage_nbytes + width * width * 4) > device.available_bytes:
                raise ValueError("matrix probe working set exceeds available capacity")
            # Exact dyadic values avoid an arbitrary relaxed numerical check.
            carrier = np.float32 if dtype == DType.BF16 else dtype.value
            values = ((np.arange(spec.elements, dtype=np.int32) % 17 - 8) / 16).astype(carrier).reshape(spec.shape)
            probe(matrix, (Argument(spec, "left"), Argument(spec, "right", ValueKind.CONSTANT)),
                  (values, values), Resource.MATRIX_ARITHMETIC, "rate:matrix-work", dtype, body=matrix_probe)
        width = protocol.vector_width
        for dtype, resource, metric in (
            (DType.F32, Resource.VECTOR_ARITHMETIC, "rate:floating-work"),
            (DType.U32, Resource.INTEGER_ARITHMETIC, "rate:integer-work"),
        ):
            spec = TensorSpec((width,), dtype)
            left = np.ones(width, dtype=dtype.value)
            right = np.ones(width, dtype=dtype.value)
            probe(arithmetic, (Argument(spec, "left"), Argument(spec, "right")),
                  (left, right), resource, metric, dtype)
        spec = TensorSpec((width,), DType.F32)
        probe(special, (Argument(spec, "value"),), (np.zeros(width, dtype=np.float32),),
              Resource.SPECIAL_FUNCTIONS, "rate:special-functions", DType.F32)
        probe(comparisons, (Argument(spec, "left"), Argument(spec, "right")),
              (np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)),
              Resource.COMPARISONS, "rate:comparisons", DType.F32)
        result = Characterization(identity=str(uuid4()), key=key, created=datetime.now(UTC),
                                  device=identity, compiler=device.compiler_identity,
                                  capabilities=device.capabilities.fingerprint, protocol=protocol,
                                  rates=tuple(rates), unavailable=tuple(unavailable))
        store.publish_characterization(result)
        return result


def characterize_sources(device, store, profile, spans, *, cancellation=None):
    """Characterize selected source paths using the ordinary streamed copy formula.

    This extends the profile without rerunning or changing its device references.
    Source reads are bounded; a memory-source rate is never reused for a file.
    """
    from .runner import MeasurementCancelled, MeasurementRunner

    known = {rate.source for rate in profile.rates if rate.source is not None}
    selected = {}
    for span in spans:
        info = span.source.info
        if info not in known and (info not in selected or selected[info].length < span.length):
            selected[info] = span
    if not selected:
        return profile
    payload = (profile.identity, "source-copy-v1", sorted(info.fingerprint for info in selected))
    key = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
    cached = store.characterization(key)
    if cached is not None:
        return cached
    rates = list(profile.rates)
    for info, span in selected.items():
        if cancellation is not None and cancellation.is_set():
            raise MeasurementCancelled("source characterization cancelled")
        size = min(span.length, 16 << 20)
        spec = TensorSpec((size,), DType.U8)
        signature = Signature((Argument(spec, "source", ValueKind.CONSTANT),
                               Argument(spec, "destination", ValueKind.RESOURCE),
                               Argument(TensorSpec((2,), DType.I64), "extent")))
        graph = trace(copy, signature)
        ids = {value.name: value.id for value in graph.values if value.producer is None}
        source = np.frombuffer(span.source.read(span.offset, size), dtype=np.uint8)
        binding = Binding(spec, f"source-probe:{info.fingerprint}:{span.offset}:{size}", Residency.STREAMED,
                          (SourcePlane(SourceSpan(span.source, span.offset, size), 1, 1),), DenseImport(DType.U8))
        fixture = Fixture(graph, {ids["source"]: source, ids["destination"]: np.zeros(size, dtype=np.uint8),
                                  ids["extent"]: np.array([0, size], dtype=np.int64)},
                          bindings={ids["source"]: binding})
        runner = MeasurementRunner(fixture, device, store, CompileOptions(mode="prefill"),
                                   protocol=profile.protocol.measurement, prepared_limit=1, _resource_probe=True)
        try:
            target, = FormulaTree(graph).roots
            measurement = runner.measure(target, cancellation=cancellation).measurement
            if measurement.outcome != Outcome.COMPLETE:
                raise RuntimeError(measurement.error or "source characterization did not complete")
            metric = next(item for item in measurement.metrics if item.name == "source-read-rate")
            rates.append(Rate(resource=Resource.SOURCE_IMPORT, value=metric.value, unit=metric.unit,
                              measurement=measurement.identity, metric=metric.name, working_set_bytes=size,
                              dtype=DType.U8, source=info, conditions=(
                                  f"source snapshot {info.fingerprint}; contiguous region {span.offset}:{span.offset + size}",
                                  "source API bytes / complete streamed-copy wall duration, including import and completion",
                                  "reference preparation reads the probe region; filesystem cache uncontrolled, not cold-disk bandwidth",
                              )))
        finally:
            runner.close()
    result = profile.model_copy(update={"identity": str(uuid4()), "key": key, "created": datetime.now(UTC),
                                        "rates": tuple(rates)})
    store.publish_characterization(result)
    return result
