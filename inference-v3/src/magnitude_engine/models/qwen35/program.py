"""Dense Qwen equations over bound logical operations and candidate state.

This program receives embeddings, logical causal positions, and rotary coordinates.
It does not tokenize, interpret images, schedule requests, or decode weight formats.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from itertools import pairwise

from magnitude_engine.models.qwen35.artifact import (
    AttentionWeights,
    DenseArtifact,
    RecurrentWeights,
)
from magnitude_engine.models.qwen35.state import QwenAdvance
from magnitude_engine.models.qwen35.workspace import Slot, Workspace
from magnitude_engine.numerics.policy import NumericalFamily
from magnitude_engine.numerics.semantics import Pointwise
from magnitude_engine.operations.activation import Elementwise, RMSNorm
from magnitude_engine.operations.attention import CausalAttention, KVAppend
from magnitude_engine.operations.copy import Copy
from magnitude_engine.operations.factory import WeightOperations
from magnitude_engine.operations.kv_binding import (
    KVBinding,
    ReadBinding,
    ReadGeometry,
    WriteBinding,
    WriteGeometry,
)
from magnitude_engine.operations.linear import Linear
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.operations.projections import Projections
from magnitude_engine.operations.recurrent import DeltaRecurrence, RecurrentPreparation
from magnitude_engine.operations.rotary import AttentionPreparation
from magnitude_engine.platform.binding import BoundSequence, InputLayout
from magnitude_engine.platform.execution import DType, Prepared, Tensor, TensorSpec


@dataclass(frozen=True)
class AttentionOperands:
    reads: tuple[ReadBinding, ...]
    writes: tuple[WriteBinding, ...]


@dataclass(frozen=True)
class InvocationGeometry:
    layout: InputLayout
    counts: tuple[int, ...]
    output_rows: tuple[int, ...]
    # Layer-independent geometry controls attention and append specializations.
    reads: tuple[tuple[ReadGeometry, ...], ...]
    writes: tuple[tuple[WriteGeometry, ...], ...]


@dataclass(frozen=True)
class AttentionMixer:
    inputs: Projections
    preparation: AttentionPreparation
    output: Linear
    state_layer: int


@dataclass(frozen=True)
class RecurrentMixer:
    inputs: Projections
    preparation: RecurrentPreparation
    norm: RMSNorm
    output: Linear


@dataclass(frozen=True)
class Block:
    input_norm: RMSNorm
    mixer: AttentionMixer | RecurrentMixer
    feedforward_norm: RMSNorm
    feedforward: Linear
    down: Linear


class DenseProgram:
    def __init__(
        self,
        description: DenseArtifact,
        operations: WeightOperations,
        numerics: NumericalFamily = NumericalFamily.NATIVE_BF16,
    ):
        if description.artifact_identity != operations.artifact_identity:
            raise ValueError("model description and weight operations refer to different artifacts")
        self.description, self.geometry, self.operations = (
            description,
            description.geometry,
            operations,
        )
        self.context, self.numerics = operations.context, numerics
        self._binding: BoundSequence | None = None
        self._binding_geometry: InvocationGeometry | None = None
        g = self.geometry
        with ExitStack() as cleanup:
            self.workspace = Workspace(self.context, numerics)
            cleanup.callback(self.workspace.close)
            self.copy = Copy(self.context)
            self.add = Elementwise(self.context, Pointwise.ADD)
            self.silu_product = Elementwise(self.context, Pointwise.SILU_PRODUCT)
            self.sigmoid_product = Elementwise(
                self.context, Pointwise.SIGMOID_PRODUCT, native_rounding=numerics.native_rounding
            )
            self.delta = DeltaRecurrence(
                self.context,
                g.recurrent_key_heads,
                g.recurrent_value_heads,
                g.recurrent_width,
                g.recurrent_head_mapping,
            )
            self.attention = CausalAttention(
                self.context, g.attention_heads, g.kv_heads, g.attention_width
            )
            self.append = KVAppend(self.context, g.kv_heads, g.attention_width)
            for operation in (
                self.copy,
                self.add,
                self.silu_product,
                self.sigmoid_product,
                self.delta,
                self.attention,
                self.append,
            ):
                cleanup.callback(operation.close)

            def norm(weight):
                value = RMSNorm(
                    self.context,
                    operations.parameter(weight),
                    g.epsilon,
                    native_rounding=numerics.native_rounding,
                )
                cleanup.callback(value.close)
                return value

            blocks = []
            attention_layer = 0
            for weights in description.blocks:
                if isinstance(weights.mixer, AttentionWeights):
                    w = weights.mixer
                    preparation = AttentionPreparation(
                        self.context,
                        operations.parameter(w.query_norm),
                        operations.parameter(w.key_norm),
                        g.attention_heads,
                        g.kv_heads,
                        g.attention_width,
                        g.rotary_width,
                        g.rotary_base,
                        g.rotary_sections,
                        g.epsilon,
                        native_rounding=numerics.native_rounding,
                    )
                    cleanup.callback(preparation.close)
                    mixer = AttentionMixer(
                        operations.projections((w.query_gate, w.key, w.value)),
                        preparation,
                        operations.linear(w.output),
                        attention_layer,
                    )
                    attention_layer += 1
                else:
                    w = weights.mixer
                    assert isinstance(w, RecurrentWeights)
                    preparation = RecurrentPreparation(
                        self.context,
                        operations.parameter(w.convolution),
                        operations.parameter(w.decay),
                        operations.parameter(w.time_bias),
                        g.recurrent_key_heads,
                        g.recurrent_value_heads,
                        g.recurrent_width,
                        g.convolution_width,
                        g.epsilon,
                        native_rounding=numerics.native_rounding,
                    )
                    cleanup.callback(preparation.close)
                    mixer = RecurrentMixer(
                        operations.projections((w.query_key_value, w.gate, w.beta, w.alpha)),
                        preparation,
                        norm(w.norm),
                        operations.linear(w.output),
                    )
                blocks.append(
                    Block(
                        norm(weights.input_norm),
                        mixer,
                        norm(weights.feedforward_norm),
                        operations.gated_linear(
                            weights.feedforward_gate,
                            weights.feedforward_up,
                            native_rounding=numerics.native_rounding,
                        ),
                        operations.linear(weights.feedforward_down),
                    )
                )
            self.blocks = tuple(blocks)
            self.output_norm = norm(description.output_norm)
            self.readout = operations.linear(description.output)
            self.embedding = operations.embedding(description.embedding)
            self._cleanup = cleanup.pop_all()

    def _release_plan(self) -> None:
        """A new operand layout does not discard reusable operation workspaces."""
        if self._binding is not None:
            self._binding.close()
            self._binding = None
            self._binding_geometry = None

    def release_binding(self) -> None:
        """Evict the plan and numerical scratch; outstanding consumers retain claims."""
        self._release_plan()
        self.attention.release_workspace()
        self.operations.release_workspace()

    def reserve(self, rows: int) -> None:
        if rows <= 0:
            raise ValueError("forward must contain input rows")
        g = self.geometry
        hidden = rows * g.hidden
        attention = rows * g.attention_heads * g.attention_width
        recurrent = rows * g.recurrent_value_heads * g.recurrent_width
        generation = self.workspace.generation
        self.workspace.reserve(
            {
                Slot.INPUT: hidden,
                Slot.HIDDEN: hidden,
                Slot.RESIDUAL: hidden,
                Slot.NORMALIZED: hidden,
                Slot.READOUT: hidden,
                Slot.READOUT_INPUT: hidden,
                Slot.MIXER_OUTPUT: hidden,
                Slot.MIXER_PROJECTIONS: rows
                * max(
                    g.recurrent_channels + g.recurrent_value_heads * (g.recurrent_width + 2),
                    2 * (g.attention_heads + g.kv_heads) * g.attention_width,
                ),
                Slot.QUERY_PREPARED: attention,
                Slot.KEY_PREPARED: rows * g.kv_heads * g.attention_width,
                Slot.GATE_PREPARED: attention,
                Slot.ATTENDED: attention,
                Slot.ATTENTION_GATED: attention,
                Slot.RECURRENT_QUERY: rows * g.recurrent_key_heads * g.recurrent_width,
                Slot.RECURRENT_KEY: rows * g.recurrent_key_heads * g.recurrent_width,
                Slot.RECURRENT_VALUE: rows * g.recurrent_value_heads * g.recurrent_width,
                Slot.RECURRENT_BETA: rows * g.recurrent_value_heads,
                Slot.RECURRENT_DECAY: rows * g.recurrent_value_heads,
                Slot.RECURRENT_MIXED: recurrent,
                Slot.RECURRENT_NORMALIZED: recurrent,
                Slot.RECURRENT_GATED: recurrent,
                Slot.FEEDFORWARD_ACTIVATED: rows * g.intermediate,
            }
        )
        if generation != self.workspace.generation:
            self.release_binding()
        self.operations.reserve_workspace(rows, self.numerics.activation)

    def input_buffer(self, rows: int) -> Tensor:
        self.reserve(rows)
        return self.workspace.acquire(Slot.INPUT, (rows, self.geometry.hidden))

    def prepare(
        self,
        embeddings: Tensor,
        coordinates: Tensor,
        positions: Tensor,
        advances: tuple[QwenAdvance, ...],
        logits: Tensor | None,
        *,
        output_rows: tuple[int, ...] = (),
    ) -> tuple[Prepared, ...]:
        g = self.geometry
        if not advances or len({id(advance.state) for advance in advances}) != len(advances):
            raise ValueError("a packed forward needs distinct sequence advances")
        ranges: list[tuple[int, QwenAdvance]] = []
        rows = 0
        for advance in advances:
            if (
                advance.closed
                or advance.state.pending is not advance
                or advance.state.store.geometry != g
                or advance.state.store.context is not self.context
            ):
                raise ValueError("forward state does not match this bound model geometry")
            ranges.append((rows, advance))
            rows += advance.count
        if (
            embeddings.spec != TensorSpec((rows, g.hidden), self.numerics.activation)
            or coordinates.spec != TensorSpec((rows, 3), DType.I32)
            or positions.spec != TensorSpec((rows,), DType.I32)
        ):
            raise ValueError("Qwen inputs differ from the forward layout")
        if (
            (logits is None) != (not output_rows)
            or any(type(row) is not int or not 0 <= row < rows for row in output_rows)
            or any(b <= a for a, b in pairwise(output_rows))
            or (
                logits is not None
                and logits.spec != TensorSpec((len(output_rows), g.vocabulary), DType.F32)
            )
        ):
            raise ValueError("logit output differs from the requested row selection")
        self.reserve(rows)
        with Preparation(self.context) as p:
            dynamic = [embeddings, coordinates, positions]
            if logits is not None:
                dynamic.append(logits)
            state_bindings: dict[QwenAdvance, tuple[AttentionOperands, ...]] = {}
            read_geometry, write_geometry = [], []
            for advance in advances:
                for recurrent in advance.recurrent:
                    if recurrent is not None:
                        dynamic.extend(
                            (
                                recurrent.previous.convolution,
                                recurrent.previous.delta,
                                recurrent.following.convolution,
                                recurrent.following.delta,
                            )
                        )
                kv = KVBinding(p, advance.reads, advance.writes)
                layers = []
                for layer in range(sum(isinstance(b.mixer, AttentionMixer) for b in self.blocks)):
                    reads, writes = kv.reads(layer), kv.appends(layer)
                    layers.append(AttentionOperands(reads, writes))
                    for read in reads:
                        dynamic.extend(read.operands)
                    for write in writes:
                        dynamic.extend(write.operands)
                state_bindings[advance] = tuple(layers)
                read_geometry.append(
                    tuple(read.geometry for read in layers[0].reads) if layers else ()
                )
                write_geometry.append(
                    tuple(write.geometry for write in layers[0].writes) if layers else ()
                )
            # Metadata shared by layers has one invocation slot. Distinct views
            # retain their explicit alias relationship in the platform layout.
            inputs = tuple(dict.fromkeys(dynamic))
            geometry = InvocationGeometry(
                InputLayout.of(inputs),
                tuple(advance.count for advance in advances),
                output_rows,
                tuple(read_geometry),
                tuple(write_geometry),
            )
            if self._binding is not None and geometry == self._binding_geometry:
                return (self._binding.prepare(inputs),)
            self._release_plan()
            with ExitStack() as cleanup:
                cleanup.callback(self.attention.release_workspace)
                commands = self._commands(
                    embeddings,
                    coordinates,
                    positions,
                    ranges,
                    logits,
                    output_rows,
                    state_bindings,
                )
                binding = BoundSequence(self.context, commands, inputs)
                cleanup.callback(binding.close)
                invocation = binding.prepare(inputs)
                self._binding, self._binding_geometry = binding, geometry
                cleanup.pop_all()
            return (invocation,)

    def _commands(
        self,
        embeddings: Tensor,
        coordinates: Tensor,
        positions: Tensor,
        ranges: list[tuple[int, QwenAdvance]],
        logits: Tensor | None,
        output_rows: tuple[int, ...],
        state_bindings: dict[QwenAdvance, tuple[AttentionOperands, ...]],
    ) -> tuple[Prepared, ...]:
        g = self.geometry
        rows = embeddings.spec.shape[0]
        with Preparation(self.context) as p:

            def view(slot, *shape, offset=0):
                return self.workspace.view(p, slot, shape, offset)

            def sequence_view(tensor: Tensor, start: int, count: int) -> Tensor:
                spec = TensorSpec((count, *tensor.spec.shape[1:]), tensor.spec.dtype)
                return p.view(tensor, spec, start * tensor.spec.nbytes // rows)

            hidden = view(Slot.HIDDEN, rows, g.hidden)
            residual = view(Slot.RESIDUAL, rows, g.hidden)
            normalized = view(Slot.NORMALIZED, rows, g.hidden)
            mixer_output = view(Slot.MIXER_OUTPUT, rows, g.hidden)
            activated = view(Slot.FEEDFORWARD_ACTIVATED, rows, g.intermediate)
            current = embeddings
            for index, block in enumerate(self.blocks):
                needs_output = logits is not None or index + 1 < len(self.blocks)
                p.add(*block.input_norm.prepare(current, normalized))
                mixer = block.mixer
                if isinstance(mixer, RecurrentMixer):
                    packed = view(Slot.MIXER_PROJECTIONS, rows * sum(mixer.inputs.widths))
                    projected, z, beta_input, alpha = mixer.inputs.outputs(p, packed, rows)
                    p.add(*mixer.inputs.prepare(normalized, packed))
                    queries = view(
                        Slot.RECURRENT_QUERY, rows, g.recurrent_key_heads, g.recurrent_width
                    )
                    keys = view(Slot.RECURRENT_KEY, rows, g.recurrent_key_heads, g.recurrent_width)
                    values = view(
                        Slot.RECURRENT_VALUE, rows, g.recurrent_value_heads, g.recurrent_width
                    )
                    beta = view(Slot.RECURRENT_BETA, rows, g.recurrent_value_heads)
                    decay = view(Slot.RECURRENT_DECAY, rows, g.recurrent_value_heads)
                    mixed = view(
                        Slot.RECURRENT_MIXED, rows, g.recurrent_value_heads, g.recurrent_width
                    )
                    for start, advance in ranges:
                        state = advance.recurrent[index]
                        if state is None:
                            raise ValueError("recurrent block has no candidate state")

                        sequence_queries = sequence_view(queries, start, advance.count)
                        sequence_keys = sequence_view(keys, start, advance.count)
                        sequence_values = sequence_view(values, start, advance.count)
                        sequence_beta = sequence_view(beta, start, advance.count)
                        sequence_decay = sequence_view(decay, start, advance.count)
                        p.add(
                            *mixer.preparation.prepare(
                                sequence_view(projected, start, advance.count),
                                sequence_view(alpha, start, advance.count),
                                sequence_view(beta_input, start, advance.count),
                                state.previous.convolution,
                                state.following.convolution,
                                sequence_queries,
                                sequence_keys,
                                sequence_values,
                                sequence_beta,
                                sequence_decay,
                            )
                        )
                        p.add(
                            *self.delta.prepare(
                                sequence_queries,
                                sequence_keys,
                                sequence_values,
                                sequence_decay,
                                sequence_beta,
                                state.previous.delta,
                                state.following.delta,
                                sequence_view(mixed, start, advance.count),
                            )
                        )
                    if not needs_output:
                        # The final state is complete. Its stateless suffix has
                        # no consumer during a state-only prefill chunk.
                        break
                    normed = view(
                        Slot.RECURRENT_NORMALIZED, rows, g.recurrent_value_heads, g.recurrent_width
                    )
                    flat = p.view(normed, z.spec)
                    gated = view(
                        Slot.RECURRENT_GATED, rows, g.recurrent_value_heads * g.recurrent_width
                    )

                    p.add(*mixer.norm.prepare(mixed, normed))
                    p.add(*self.silu_product.prepare(z, flat, gated))
                    p.add(*mixer.output.prepare(gated, mixer_output))
                else:
                    packed = view(Slot.MIXER_PROJECTIONS, rows * sum(mixer.inputs.widths))
                    projected, raw_keys, raw_values = mixer.inputs.outputs(p, packed, rows)
                    p.add(*mixer.inputs.prepare(normalized, packed))
                    queries = view(Slot.QUERY_PREPARED, rows, g.attention_heads, g.attention_width)
                    keys = view(Slot.KEY_PREPARED, rows, g.kv_heads, g.attention_width)
                    gate_values = view(
                        Slot.GATE_PREPARED, rows, g.attention_heads, g.attention_width
                    )
                    p.add(
                        *mixer.preparation.prepare(
                            projected, raw_keys, coordinates, queries, keys, gate_values
                        )
                    )
                    values = p.view(raw_values, keys.spec)
                    attended = view(Slot.ATTENDED, rows, g.attention_heads, g.attention_width)
                    gated = view(Slot.ATTENTION_GATED, rows, g.attention_heads, g.attention_width)
                    for start, advance in ranges:
                        p.add(
                            *self.append.prepare(
                                sequence_view(keys, start, advance.count),
                                sequence_view(values, start, advance.count),
                                state_bindings[advance][mixer.state_layer].writes,
                            )
                        )
                        if needs_output:
                            p.add(
                                *self.attention.prepare(
                                    sequence_view(queries, start, advance.count),
                                    sequence_view(positions, start, advance.count),
                                    state_bindings[advance][mixer.state_layer].reads,
                                    sequence_view(attended, start, advance.count),
                                )
                            )
                    if not needs_output:
                        break
                    flat = p.view(
                        gated,
                        TensorSpec(
                            (rows, g.attention_heads * g.attention_width), self.numerics.activation
                        ),
                    )

                    p.add(*self.sigmoid_product.prepare(attended, gate_values, gated))
                    p.add(*mixer.output.prepare(flat, mixer_output))
                p.add(*self.add.prepare(current, mixer_output, residual))

                p.add(*block.feedforward_norm.prepare(residual, normalized))
                p.add(*block.feedforward.prepare(normalized, activated))
                p.add(*block.down.prepare(activated, mixer_output))
                p.add(*self.add.prepare(residual, mixer_output, hidden))
                current = hidden
            if logits is not None:
                # Select requested residual rows first. Normalizing an entire
                # prefill chunk for a single output wastes work and couples the
                # readout's precision to the internal matrix-input workspace.
                selected_spec = TensorSpec((len(output_rows), g.hidden), current.spec.dtype)
                row_spec = TensorSpec((1, g.hidden), current.spec.dtype)
                if output_rows == tuple(range(output_rows[0], output_rows[0] + len(output_rows))):
                    selected = p.view(current, selected_spec, output_rows[0] * row_spec.nbytes)
                else:
                    selected = view(Slot.READOUT_INPUT, len(output_rows), g.hidden)
                    for destination, source in enumerate(output_rows):
                        p.add(
                            *self.copy.prepare(
                                p.view(current, row_spec, source * row_spec.nbytes),
                                p.view(selected, row_spec, destination * row_spec.nbytes),
                            )
                        )
                readout = view(Slot.READOUT, len(output_rows), g.hidden)
                p.add(*self.output_norm.prepare(selected, readout))
                p.add(*self.readout.prepare(readout, logits))
            return p.finish()

    def close(self) -> None:
        self.release_binding()
        self._cleanup.close()
