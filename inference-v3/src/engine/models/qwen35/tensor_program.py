"""Whole-step Qwen specialization over Ops resources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import ops
from . import operations as _operations
from engine.weights.residency import WeightResidency
from engine.models.qwen35.description import (
    AttentionWeights,
    DenseDescription,
    DenseFeedForwardWeights,
    RecurrentWeights,
    RoutedFeedForwardWeights,
)
from engine.models.qwen35.equations import (
    AttentionTensors,
    BlockTensors,
    DecoderTensors,
    DenseFeedForwardTensors,
    RecurrentTensors,
    RoutedFeedForwardTensors,
    decoder,
)


@dataclass(frozen=True, slots=True)
class InvocationSpecs:
    batch: int
    tokens: ops.TensorSpec
    coordinates: ops.TensorSpec
    recurrent_offsets: ops.TensorSpec | None
    output_rows: ops.TensorSpec | None
    draws: ops.TensorSpec | None
    destinations: tuple[ops.TensorSpec, ...]
    visible: tuple[ops.TensorSpec, ...]
    attention_state: tuple[ops.TensorSpec, ...]
    convolution_state: tuple[ops.TensorSpec, ...]
    delta_state: tuple[ops.TensorSpec, ...]
    features: tuple[ops.TensorSpec, ...] = ()
    feature_rows: tuple[ops.TensorSpec, ...] = ()

    def __post_init__(self) -> None:
        if type(self.batch) is not int or self.batch <= 0:
            raise ValueError("Qwen invocation batch must be positive")
        if self.tokens.rank != 1 or not self.tokens.dtype.integer:
            raise ValueError("Qwen invocation tokens must be a one-dimensional integer tensor")
        if self.coordinates != ops.TensorSpec((self.tokens.shape[0], 3), ops.DType.I32):
            raise ValueError("Qwen coordinates differ from token rows")
        if self.recurrent_offsets is not None and self.recurrent_offsets != ops.TensorSpec(
            (self.batch + 1,), ops.DType.I32
        ):
            raise ValueError("Qwen recurrent offsets differ from the sequence batch")
        if self.output_rows is not None and (
            self.output_rows.rank != 1 or not self.output_rows.dtype.integer
        ):
            raise ValueError("Qwen output rows must be one-dimensional integer indices")
        output_rows = self.output_rows
        if (output_rows is None) != (self.draws is None) or (
            output_rows is not None
            and self.draws != ops.TensorSpec((output_rows.shape[0], 6), ops.DType.U32)
        ):
            raise ValueError("Qwen draw rows must exactly match selected output rows")
        if len(self.features) != len(self.feature_rows) or any(
            value.rank != 2
            or value.dtype != ops.DType.F32
            or rows != ops.TensorSpec((value.shape[0],), ops.DType.I32)
            for value, rows in zip(self.features, self.feature_rows, strict=True)
        ):
            raise ValueError("Qwen feature values and row indices must be paired")


@dataclass(frozen=True, slots=True)
class ProgramDefinition:
    function: Callable[..., Any]
    signature: ops.Signature
    options: ops.CompileOptions
    constant_specs: Mapping[str, ops.TensorSpec]

    def fixture(self, values, *, bindings=None, capture=None):
        """Prepare Lab boundaries from this exact production definition.

        Values use the existing invocation argument names. The optional capture
        receives the actual typed root Value for lazily supplied artifact data.
        No GPU is opened and no alternative model/benchmark equation is defined.
        """
        from ops.lab import Fixture

        graph = ops.trace(self.function, self.signature)
        declared = {graph.value(identity).name: identity
                    for identity in (*graph.inputs, *graph.constants, *graph.resources)}
        unknown = (set(values) | set(bindings or {})) - declared.keys()
        if unknown:
            raise ValueError(f"fixture contains unknown invocation arguments: {sorted(unknown)}")
        return Fixture.from_inputs(
            graph, {declared[name]: value for name, value in values.items()},
            bindings={declared[name]: value for name, value in (bindings or {}).items()},
            capture=capture,
        )


class TensorProgram:
    """Description-bound factory for maximal Ops specializations."""

    def __init__(
        self,
        description: DenseDescription,
        device: ops.DeviceRuntime,
        weights: Mapping[str, ops.Resource | ops.Binding] | WeightResidency,
    ):
        self.description = description
        self.device = device
        expected = tuple(weight_roles(description))
        self.weights = (
            dict(weights)
            if isinstance(weights, Mapping)
            else {
                descriptor.name: weights.bind(descriptor, dtype)
                for descriptor, dtype in expected
            }
        )
        missing = [weight.name for weight, _ in expected if weight.name not in self.weights]
        if missing:
            raise KeyError(f"missing Qwen weights: {missing}")
        for descriptor, _ in expected:
            binding = self.weights[descriptor.name]
            if binding.spec.shape != descriptor.shape or (
                isinstance(binding, ops.Resource) and binding.device is not device
            ):
                raise ValueError(f"weight {descriptor.name!r} has incompatible geometry or ownership")
        self._compiled: dict[tuple[str, InvocationSpecs, str], ops.CompiledFunction] = {}

    def specialize(
        self,
        mode: str,
        specs: InvocationSpecs,
        *,
        static_resources: Mapping[int | str, ops.Resource] | None = None,
        precision: str = "model",
    ) -> ops.CompiledFunction:
        key = mode, specs, precision
        compiled = self._compiled.get(key)
        if compiled is not None:
            return compiled
        definition = define(
            self.description,
            {name: resource.spec for name, resource in self.weights.items()},
            mode,
            specs,
            precision=precision,
        )
        compiled = ops.compile(
            definition.function,
            signature=definition.signature,
            device=self.device,
            constants=cast(Mapping[int | str, ops.Resource | ops.Binding], self.weights),
            static_resources=static_resources,
            options=definition.options,
        )
        self._compiled[key] = compiled
        return compiled

    def close(self) -> None:
        for compiled in reversed(tuple(self._compiled.values())):
            compiled.close()
        self._compiled.clear()


def define(
    description: DenseDescription,
    weight_specs: Mapping[str, ops.TensorSpec],
    mode: str,
    specs: InvocationSpecs,
    *,
    precision: str = "model",
) -> ProgramDefinition:
    """Build Qwen's semantic function and ABI without compiling or allocating it."""
    if mode not in {"prefill", "decode"}:
        raise ValueError("Qwen specialization mode must be prefill or decode")
    if any(value.shape[1] != description.geometry.hidden for value in specs.features):
        raise ValueError("Qwen feature width differs from the decoder hidden width")
    attention = sum(isinstance(layer.mixer, AttentionWeights) for layer in description.blocks)
    recurrent = len(description.blocks) - attention
    if not (
        len(specs.destinations) == len(specs.visible) == len(specs.attention_state) == attention
        and len(specs.convolution_state) == len(specs.delta_state) == recurrent * specs.batch
    ):
        raise ValueError("Qwen invocation state does not match the decoder topology")
    if (specs.recurrent_offsets is None) != (recurrent == 0):
        raise ValueError("Qwen recurrent offsets must exactly accompany recurrent layers")
    expected = {descriptor.name for descriptor, _ in weight_roles(description)}
    if missing := sorted(expected - set(weight_specs)):
        raise KeyError(f"missing Qwen weight specifications: {missing}")
    kwargs: dict[str, ops.Argument] = {}
    for name, spec in _state_arguments(specs):
        kwargs[name] = ops.Argument(spec, name, ops.ValueKind.RESOURCE)
    for descriptor, _ in weight_roles(description):
        kwargs[descriptor.name] = ops.Argument(
            weight_specs[descriptor.name], descriptor.name, ops.ValueKind.CONSTANT
        )
    arguments = [
        ops.Argument(specs.tokens, "tokens"),
        ops.Argument(specs.coordinates, "coordinates"),
    ]
    if specs.recurrent_offsets is not None:
        arguments.append(ops.Argument(specs.recurrent_offsets, "recurrent_offsets"))
    if specs.output_rows is not None:
        arguments.append(ops.Argument(specs.output_rows, "output_rows"))
        arguments.append(ops.Argument(cast(ops.TensorSpec, specs.draws), "draws"))
    arguments.extend(
        ops.Argument(spec, f"feature.{index}.rows") for index, spec in enumerate(specs.feature_rows)
    )

    def function(*args, **bound):
        tokens, coordinates = args[:2]
        cursor = 2
        recurrent_offsets = args[cursor] if specs.recurrent_offsets is not None else None
        cursor += specs.recurrent_offsets is not None
        output_rows = args[cursor] if specs.output_rows is not None else None
        cursor += specs.output_rows is not None
        draws = args[cursor] if specs.draws is not None else None
        cursor += specs.draws is not None
        feature_rows = tuple(args[cursor:])
        feature_values = tuple(
            bound[f"feature.{index}.values"] for index in range(len(specs.features))
        )
        weights = _bind_weights(description, bound)
        convolution = tuple(
            _concatenate_batch(
                tuple(
                    bound[f"recurrent.{layer}.{sequence}.convolution"]
                    for sequence in range(specs.batch)
                )
            )
            for layer in range(recurrent)
        )
        delta = tuple(
            _concatenate_batch(
                tuple(
                    bound[f"recurrent.{layer}.{sequence}.delta"] for sequence in range(specs.batch)
                )
            )
            for layer in range(recurrent)
        )
        result = decoder(
            tokens,
            coordinates,
            tuple(bound[f"attention.{i}.destinations"] for i in range(len(specs.destinations))),
            tuple(bound[f"attention.{i}.visible"] for i in range(len(specs.visible))),
            tuple(bound[f"attention.{i}.state"] for i in range(len(specs.attention_state))),
            convolution,
            delta,
            recurrent_offsets,
            weights,
            sequence_count=specs.batch,
            output_rows=output_rows,
            feature_values=(ops.concatenate(feature_values, axis=0) if feature_values else None),
            feature_rows=(ops.concatenate(feature_rows, axis=0) if feature_rows else None),
        )
        if draws is None:
            return result
        outputs = cast(tuple[ops.Tensor, ...], result)
        return outputs[0], ops.sample(outputs[0], draws), *outputs[1:]

    return ProgramDefinition(
        function,
        ops.Signature(tuple(arguments), kwargs),
        ops.CompileOptions(mode=mode, precision=precision),
        {name: weight_specs[name] for name in expected},
    )


def _concatenate_batch(values: tuple[ops.Tensor, ...]) -> ops.Tensor:
    return values[0] if len(values) == 1 else ops.concatenate(values, axis=0)


def _state_arguments(specs: InvocationSpecs):
    for index, spec in enumerate(specs.features):
        yield f"feature.{index}.values", spec
    for index, spec in enumerate(specs.destinations):
        yield f"attention.{index}.destinations", spec
    for index, spec in enumerate(specs.visible):
        yield f"attention.{index}.visible", spec
    for index, spec in enumerate(specs.attention_state):
        yield f"attention.{index}.state", spec
    for index, spec in enumerate(specs.convolution_state):
        layer, sequence = divmod(index, specs.batch)
        yield f"recurrent.{layer}.{sequence}.convolution", spec
    for index, spec in enumerate(specs.delta_state):
        layer, sequence = divmod(index, specs.batch)
        yield f"recurrent.{layer}.{sequence}.delta", spec


def weight_roles(description: DenseDescription):
    activation = description.geometry.activation_dtype
    parameter = ops.DType.F32
    yield description.embedding, activation
    for layer in description.blocks:
        yield layer.input_norm, activation
        mixer = layer.mixer
        if isinstance(mixer, AttentionWeights):
            yield from (
                (mixer.query_gate, activation),
                (mixer.key, activation),
                (mixer.value, activation),
                (mixer.query_norm, parameter),
                (mixer.key_norm, parameter),
                (mixer.output, activation),
            )
        else:
            yield from (
                (mixer.query_key_value, activation),
                (mixer.gate, activation),
                (mixer.beta, activation),
                (mixer.alpha, activation),
                (mixer.convolution, parameter),
                (mixer.decay, parameter),
                (mixer.time_bias, parameter),
                (mixer.norm, activation),
                (mixer.output, activation),
            )
        yield layer.feedforward_norm, activation
        feedforward = layer.feedforward
        if isinstance(feedforward, DenseFeedForwardWeights):
            yield from (
                (feedforward.gate, activation),
                (feedforward.up, activation),
                (feedforward.down, activation),
            )
        else:
            yield from (
                (feedforward.router, activation),
                (feedforward.shared_router, parameter),
                (feedforward.expert_gate, activation),
                (feedforward.expert_up, activation),
                (feedforward.expert_down, activation),
                (feedforward.shared_gate, activation),
                (feedforward.shared_up, activation),
                (feedforward.shared_down, activation),
            )
    yield description.output_norm, activation
    yield description.output, activation


def _bind_weights(description: DenseDescription, values: Mapping[str, ops.Tensor]) -> DecoderTensors:
    blocks = []
    g = description.geometry
    for layer in description.blocks:
        mixer = layer.mixer
        if isinstance(mixer, AttentionWeights):
            bound_mixer = AttentionTensors(
                values[mixer.query_gate.name],
                values[mixer.key.name],
                values[mixer.value.name],
                values[mixer.query_norm.name],
                values[mixer.key_norm.name],
                values[mixer.output.name],
                g.attention_heads,
                g.kv_heads,
                g.attention_width,
                g.rotary_width,
                g.rotary_base,
                g.rotary_sections,
                g.epsilon,
            )
        else:
            assert isinstance(mixer, RecurrentWeights)
            bound_mixer = RecurrentTensors(
                values[mixer.query_key_value.name],
                values[mixer.gate.name],
                values[mixer.beta.name],
                values[mixer.alpha.name],
                values[mixer.convolution.name],
                values[mixer.decay.name],
                values[mixer.time_bias.name],
                values[mixer.norm.name],
                values[mixer.output.name],
                g.recurrent_key_heads,
                g.recurrent_value_heads,
                g.recurrent_width,
                g.convolution_width,
                g.epsilon,
                g.recurrent_head_mapping.value,
            )
        feedforward = layer.feedforward
        if isinstance(feedforward, DenseFeedForwardWeights):
            bound_feedforward = DenseFeedForwardTensors(
                values[feedforward.gate.name],
                values[feedforward.up.name],
                values[feedforward.down.name],
            )
        else:
            assert isinstance(feedforward, RoutedFeedForwardWeights)
            experts = g.experts
            if experts is None:
                raise ValueError("routed feed-forward has no expert geometry")
            bound_feedforward = RoutedFeedForwardTensors(
                values[feedforward.router.name],
                values[feedforward.shared_router.name],
                values[feedforward.expert_gate.name],
                values[feedforward.expert_up.name],
                values[feedforward.expert_down.name],
                DenseFeedForwardTensors(
                    values[feedforward.shared_gate.name],
                    values[feedforward.shared_up.name],
                    values[feedforward.shared_down.name],
                ),
                experts.selected,
                experts.normalize_selected,
            )
        blocks.append(
            BlockTensors(
                values[layer.input_norm.name],
                bound_mixer,
                values[layer.feedforward_norm.name],
                bound_feedforward,
                g.epsilon,
            )
        )
    return DecoderTensors(
        values[description.embedding.name],
        tuple(blocks),
        values[description.output_norm.name],
        values[description.output.name],
        g.epsilon,
    )
