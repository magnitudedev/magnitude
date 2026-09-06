"""Gated-delta recurrence with explicit prepared inputs and prefix reconciliation."""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from .inputs import DeltaInputs


@cache
def _kernel(state_only: bool) -> Any:
    # A SIMD group owns one state vector. State stays in registers over the
    # sequence, and only the final state is written. Reconciliation omits q/y.
    return mx.fast.metal_kernel(
        name="magnitude_delta_state" if state_only else "magnitude_delta_forward",
        input_names=["q", "k", "v", "decay", "beta", "initial", "length"],
        output_names=["final"] if state_only else ["final", "output"],
        source="""
        uint lane = thread_position_in_grid.x;
        uint channel = thread_position_in_grid.y;
        uint owner = thread_position_in_grid.z;
        if (channel >= DV) return;
        uint batch = owner / HV;
        uint head = owner % HV;
        uint key_head = head / (HV / HK);
        constexpr uint WIDTH = DK / 32;
        const uint tokens = SHORT_T > 0 ? SHORT_T : length[0];
        float memory[WIDTH];
        size_t base = (size_t(owner) * DV + channel) * DK;
        for (uint i = 0; i < WIDTH; ++i)
            memory[i] = initial[base + lane * WIDTH + i];
        // Resolve each row/head once. Advancing constant strides avoids wide
        // per-token address products in the register-resident recurrence loop.
        auto next_key = k + (size_t(batch) * tokens * HK + key_head) * DK;
        auto next_value = v + (size_t(batch) * tokens * HV + head) * DV + channel;
        auto next_decay = decay + size_t(batch) * tokens * HV + head;
        auto next_beta = beta + size_t(batch) * tokens * HV + head;
        QUERY_POINTER
        OUTPUT_POINTER
        for (uint t = 0; t < tokens; ++t) {
            float remembered = 0.0f;
            for (uint i = 0; i < WIDTH; ++i) {
                memory[i] *= float(*next_decay);
                remembered += memory[i] * float(next_key[lane * WIDTH + i]);
            }
            remembered = simd_sum(remembered);
            float residual = (float(*next_value) - remembered) * float(*next_beta);
            float answer = 0.0f;
            for (uint i = 0; i < WIDTH; ++i) {
                uint coordinate = lane * WIDTH + i;
                memory[i] += residual * float(next_key[coordinate]);
                COMPUTE_ANSWER
            }
            WRITE_ANSWER
            next_key += HK * DK;
            next_value += HV * DV;
            next_decay += HV;
            next_beta += HV;
            ADVANCE_QUERY
            ADVANCE_OUTPUT
        }
        for (uint i = 0; i < WIDTH; ++i)
            final[base + lane * WIDTH + i] = memory[i];
        """.replace(
            "COMPUTE_ANSWER",
            "" if state_only else "answer += memory[i] * float(next_query[coordinate]);",
        )
        .replace(
            "WRITE_ANSWER",
            ""
            if state_only
            else "answer = simd_sum(answer); if (lane == 0) *next_output = In(answer);",
        )
        .replace(
            "QUERY_POINTER",
            ""
            if state_only
            else "auto next_query = q + (size_t(batch) * tokens * HK + key_head) * DK;",
        )
        .replace(
            "OUTPUT_POINTER",
            ""
            if state_only
            else "auto next_output = output + (size_t(batch) * tokens * HV + head) * DV + channel;",
        )
        .replace(
            "ADVANCE_QUERY",
            "" if state_only else "next_query += HK * DK;",
        )
        .replace(
            "ADVANCE_OUTPUT",
            "" if state_only else "next_output += HV * DV;",
        ),
    )


class MetalDelta:
    """Native recurrence; long lengths normally share one runtime-count kernel.

    Explicit prefill specialization exists for controlled kernel comparisons. Short
    verification blocks keep their fixed-width variants in either configuration.
    """

    def __init__(self, *, specialize_prefill: bool = False):
        self.specialize_prefill = specialize_prefill

    def _run(self, inputs: DeltaInputs, state: mx.array, state_only: bool):
        batch, tokens, hk, dk = inputs.keys.shape
        hv, dv = inputs.values.shape[2:]
        if (
            tokens < 1
            or dk % 32
            or hv % hk
            or state.dtype != mx.float32
            or state.shape != (batch, hv, dv, dk)
            or inputs.queries.shape != inputs.keys.shape
            or inputs.values.shape[:2] != (batch, tokens)
            or inputs.decay.shape != (batch, tokens, hv)
            or inputs.beta.shape != (batch, tokens, hv)
        ):
            raise ValueError("unsupported gated-delta geometry")
        return _kernel(state_only)(
            inputs=[
                inputs.queries,
                inputs.keys,
                inputs.values,
                inputs.decay,
                inputs.beta,
                state,
                mx.array([tokens], mx.int32),
            ],
            template=[
                ("In", inputs.queries.dtype),
                ("DK", dk),
                ("DV", dv),
                ("HK", hk),
                ("HV", hv),
                ("SHORT_T", tokens if tokens <= 8 or self.specialize_prefill else 0),
            ],
            grid=(32, dv, batch * hv),
            threadgroup=(32, 4, 1),
            output_shapes=[state.shape] if state_only else [state.shape, inputs.values.shape],
            output_dtypes=[mx.float32] if state_only else [mx.float32, inputs.queries.dtype],
        )

    def advance(self, inputs: DeltaInputs, state: mx.array) -> tuple[mx.array, mx.array]:
        final, output = self._run(inputs, state, False)
        return output, final

    def reconcile(self, inputs: DeltaInputs, state: mx.array, count: int) -> mx.array:
        if not 0 <= count <= inputs.length:
            raise ValueError("invalid recurrence prefix")
        return state if count == 0 else self._run(inputs.prefix(count), state, True)[0]
