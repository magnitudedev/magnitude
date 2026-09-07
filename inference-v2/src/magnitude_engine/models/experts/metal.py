"""Selected-expert execution with reusable affine inputs and fused epilogues.

Input values and bias sums are prepared once per row. Gate/up emits prepared
activations directly for down/reduction, preserving MLX's BF16 boundaries.
"""

from functools import cache
from typing import TYPE_CHECKING, Any

import mlx.core as mx

if TYPE_CHECKING:
    from .computation import ExpertWeights


_HEADER = """
template <typename T, int BITS, int PACK>
float magnitude_load(const device T* x, thread float* values) {
    float sum = 0.0f;
    if (BITS == 4) {
        for (int i = 0; i < PACK; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            values[i] = float(x[i]);
            values[i + 1] = float(x[i + 1]) / 16.0f;
            values[i + 2] = float(x[i + 2]) / 256.0f;
            values[i + 3] = float(x[i + 3]) / 4096.0f;
        }
    } else {
        for (int i = 0; i < PACK; ++i) { sum += x[i]; values[i] = float(x[i]); }
    }
    return sum;
}

template <int BITS, int PACK>
float magnitude_dot(const device uint* weights, const thread float* x,
                    float scale, float bias, float sum) {
    float value = 0.0f;
    if (BITS == 4) {
        const device ushort* w = reinterpret_cast<const device ushort*>(weights);
        for (int i = 0; i < PACK / 4; ++i) {
            value += (x[4 * i] * (w[i] & 0x000f)
                + x[4 * i + 1] * (w[i] & 0x00f0)
                + x[4 * i + 2] * (w[i] & 0x0f00)
                + x[4 * i + 3] * (w[i] & 0xf000));
        }
    } else {
        const device uchar* w = reinterpret_cast<const device uchar*>(weights);
        for (int i = 0; i < PACK; ++i) value += x[i] * w[i];
    }
    return scale * value + sum * bias;
}
"""


@cache
def _prepare() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_affine_input",
        header=_HEADER,
        input_names=["x"],
        output_names=["values", "sums"],
        source="""
        uint pack = thread_position_in_grid.x;
        if (pack * PACK >= SIZE) return;
        float v[PACK];
        sums[pack] = magnitude_load<T, BITS, PACK>(x + pack * PACK, v);
        for (uint i = 0; i < PACK; ++i) values[pack * PACK + i] = v[i];
        """,
    )


def _prepare_input(hidden: mx.array, bits: int) -> tuple[mx.array, mx.array]:
    pack = 64 // bits
    values, sums = _prepare()(
        inputs=[hidden],
        template=[("T", hidden.dtype), ("BITS", bits), ("PACK", pack), ("SIZE", hidden.size)],
        grid=(hidden.size // pack, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[hidden.shape, (*hidden.shape[:-1], hidden.shape[-1] // pack)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return values, sums


@cache
def _gate_up() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_expert_gate_up",
        header=_HEADER,
        input_names=[
            "x",
            "x_sums",
            "inds",
            "wg",
            "sg_",
            "bg_",
            "wu",
            "su_",
            "bu_",
            "wsg",
            "ssg",
            "bsg",
            "wsu",
            "ssu",
            "bsu",
        ],
        output_names=["out", "prepared", "prepared_sums"],
        source="""
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint first = threadgroup_position_in_grid.y * ROWS + sg * 4;
        uint slot = threadgroup_position_in_grid.z;
        constexpr uint SLOTS = TOPK + SHARED;
        uint row = slot / SLOTS;
        bool shared = SHARED && slot % SLOTS == TOPK;
        uint expert = shared ? 0 : inds[row * TOPK + slot % SLOTS];
        const device uint* gate_w = shared ? wsg : wg;
        const device uint* up_w = shared ? wsu : wu;
        const device T* gate_s = shared ? ssg : sg_;
        const device T* gate_b = shared ? bsg : bg_;
        const device T* up_s = shared ? ssu : su_;
        const device T* up_b = shared ? bsu : bu_;
        constexpr uint KW = K * BITS / 32, KG = K / GROUP;
        float gate[4] = {0}, up[4] = {0};
        threadgroup T activated[ROWS];
        for (uint k = lane * PACK; k < K; k += 32 * PACK) {
            float values[PACK];
            float sum = x_sums[(row * K + k) / PACK];
            for (uint i = 0; i < PACK; ++i) values[i] = x[row * K + k + i];
            for (uint r = 0; r < 4; ++r) {
                size_t index = (size_t(expert) * N + first + r);
                size_t g = index * KG + k / GROUP;
                gate[r] += magnitude_dot<BITS, PACK>(gate_w + index * KW + k * BITS / 32,
                    values, float(gate_s[g]), float(gate_b[g]), sum);
                up[r] += magnitude_dot<BITS, PACK>(up_w + index * KW + k * BITS / 32,
                    values, float(up_s[g]), float(up_b[g]), sum);
            }
        }
        for (uint r = 0; r < 4; ++r) {
            float g = simd_sum(gate[r]), u = simd_sum(up[r]);
            if (lane == 0) {
                T gr = T(g), ur = T(u);
                T e = T(metal::exp(metal::abs(float(gr))));
                auto y = 1 / (1 + e);
                T sigmoid = T(gr < 0 ? y : 1 - y);
                T value = T(T(gr * sigmoid) * ur);
                if (!PREPARE) out[slot * N + first + r] = value;
                activated[sg * 4 + r] = value;
            }
        }
        if (PREPARE) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0 && lane == 0) {
                float sum = 0;
                for (uint i = 0; i < ROWS; i += 4) {
                    if (DOWN_BITS == 4) {
                        sum += activated[i] + activated[i+1] + activated[i+2] + activated[i+3];
                    } else {
                        for (uint j = 0; j < 4; ++j) sum += activated[i+j];
                    }
                    for (uint j = 0; j < 4; ++j) {
                        float value = float(activated[i+j]);
                        if (DOWN_BITS == 4) value /= float(1u << (4 * j));
                        prepared[slot * N + first + i + j] = value;
                    }
                }
                prepared_sums[(slot * N + first) / ROWS] = sum;
            }
        }
        """,
    )


@cache
def _down() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_expert_down_sum",
        header=_HEADER,
        input_names=[
            "x",
            "x_sums",
            "inds",
            "scores",
            "w",
            "scales",
            "biases",
            "wsh",
            "ssh",
            "bsh",
            "shared_score",
        ],
        output_names=["out"],
        source="""
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint first = threadgroup_position_in_grid.y * 8 + sg * 4;
        uint row = threadgroup_position_in_grid.z;
        constexpr uint KW = K * BITS / 32, KG = K / GROUP;
        float combined[4] = {0};
        constexpr uint SLOTS = TOPK + SHARED;
        for (uint slot = 0; slot < SLOTS; ++slot) {
            bool shared = SHARED && slot == TOPK;
            uint expert = shared ? 0 : inds[row * TOPK + slot];
            const device uint* selected_w = shared ? wsh : w;
            const device T* selected_s = shared ? ssh : scales;
            const device T* selected_b = shared ? bsh : biases;
            T score = shared ? shared_score[row] : scores[row * TOPK + slot];
            float acc[4] = {0};
            for (uint k = lane * PACK; k < K; k += 32 * PACK) {
                float values[PACK];
                size_t offset = (row * SLOTS + slot) * K + k;
                float sum = x_sums[offset / PACK];
                for (uint i = 0; i < PACK; ++i) values[i] = x[offset + i];
                for (uint r = 0; r < 4; ++r) {
                    size_t index = size_t(expert) * N + first + r;
                    size_t g = index * KG + k / GROUP;
                    acc[r] += magnitude_dot<BITS, PACK>(selected_w + index * KW + k * BITS / 32,
                        values, float(selected_s[g]), float(selected_b[g]), sum);
                }
            }
            for (uint r = 0; r < 4; ++r) {
                float value = simd_sum(acc[r]);
                // MLX's short column reduction adds in the output dtype.
                T contribution = T(T(value) * score);
                combined[r] = float(T(combined[r] + float(contribution)));
            }
        }
        if (lane == 0) for (uint r = 0; r < 4; ++r)
            out[row * N + first + r] = T(combined[r]);
        """,
    )


def supported(weights: "ExpertWeights", hidden: mx.array, assignments: mx.array) -> bool:
    gate, up, down = weights.gate, weights.up, weights.down
    if (
        hidden.size // hidden.shape[-1] > 8
        # Sorted upstream gathers use a different matrix reduction boundary.
        or assignments.size >= 64
        or assignments.shape[-1] > 16
        or hidden.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or gate.encoding != up.encoding
        or gate.weight.shape != up.weight.shape
    ):
        return False
    for projection in (gate, up, down):
        if (
            projection.encoding.bits not in (4, 8)
            or projection.encoding.group_size % 8
            or projection.scales.dtype != hidden.dtype
            or projection.biases.dtype != hidden.dtype
            or projection.weight.dtype != mx.uint32
            or projection.weight.ndim != 3
        ):
            return False
        width = projection.weight.shape[-1] * 32 // projection.encoding.bits
        per_lane = 64 // projection.encoding.bits
        if (
            width % (32 * per_lane)
            or projection.encoding.group_size % per_lane
            or width % projection.encoding.group_size
            or projection.weight.shape[-2] % 8
        ):
            return False
    return True


def shared_supported(
    weights: "ExpertWeights", shared: "ExpertWeights", hidden: mx.array, assignments: mx.array
) -> bool:
    return supported(weights, hidden, assignments) and all(
        a.encoding == b.encoding
        and a.weight.shape[1:] == b.weight.shape
        and a.scales.shape[1:] == b.scales.shape
        and a.biases.shape[1:] == b.biases.shape
        and a.weight.dtype == b.weight.dtype
        and a.scales.dtype == b.scales.dtype
        and a.biases.dtype == b.biases.dtype
        for a, b in zip(
            (weights.gate, weights.up, weights.down),
            (shared.gate, shared.up, shared.down),
            strict=True,
        )
    )


def _activate(
    weights: "ExpertWeights",
    hidden: mx.array,
    assignments: mx.array,
    shared: "ExpertWeights | None",
    *,
    prepare_down: bool,
) -> tuple[mx.array, ...]:
    gate, up, down = weights.gate, weights.up, weights.down
    rows, width = hidden.size // hidden.shape[-1], hidden.shape[-1]
    top_k, intermediate = assignments.shape[-1], gate.weight.shape[1]
    slots = top_k + (shared is not None)
    tile = 64 // down.encoding.bits if prepare_down else 8
    sg, su = (shared.gate, shared.up) if shared is not None else (gate, up)
    x, sums = _prepare_input(hidden, gate.encoding.bits)
    outputs = _gate_up()(
        inputs=[
            x,
            sums,
            assignments.reshape(rows, top_k),
            gate.weight,
            gate.scales,
            gate.biases,
            up.weight,
            up.scales,
            up.biases,
            sg.weight,
            sg.scales,
            sg.biases,
            su.weight,
            su.scales,
            su.biases,
        ],
        template=[
            ("T", hidden.dtype),
            ("K", width),
            ("N", intermediate),
            ("TOPK", top_k),
            ("SHARED", shared is not None),
            ("ROWS", tile),
            ("PREPARE", prepare_down),
            ("DOWN_BITS", down.encoding.bits),
            ("BITS", gate.encoding.bits),
            ("GROUP", gate.encoding.group_size),
            ("PACK", 64 // gate.encoding.bits),
        ],
        grid=(tile * 8, intermediate // tile, rows * slots),
        threadgroup=(tile * 8, 1, 1),
        output_shapes=[
            (1,) if prepare_down else (rows, slots, intermediate),
            (rows, slots, intermediate) if prepare_down else (1,),
            (rows, slots, intermediate // tile) if prepare_down else (1,),
        ],
        output_dtypes=[hidden.dtype, mx.float32, mx.float32],
    )
    return tuple(outputs)


def activate(
    weights: "ExpertWeights",
    hidden: mx.array,
    assignments: mx.array,
    shared: "ExpertWeights | None" = None,
) -> mx.array:
    return _activate(weights, hidden, assignments, shared, prepare_down=False)[0]


def apply(
    weights: "ExpertWeights",
    hidden: mx.array,
    assignments: mx.array,
    scores: mx.array,
    *,
    shared: "ExpertWeights | None" = None,
    shared_score: mx.array | None = None,
) -> mx.array:
    down = weights.down
    rows, width = hidden.size // hidden.shape[-1], hidden.shape[-1]
    top_k, intermediate = assignments.shape[-1], weights.gate.weight.shape[1]
    _, activation, sums = _activate(weights, hidden, assignments, shared, prepare_down=True)
    sd = shared.down if shared is not None else down
    if shared is not None and shared_score is None:
        raise ValueError("shared expert requires its coefficient")
    return _down()(
        inputs=[
            activation,
            sums,
            assignments.reshape(rows, top_k),
            scores.reshape(rows, top_k),
            down.weight,
            down.scales,
            down.biases,
            sd.weight,
            sd.scales,
            sd.biases,
            shared_score if shared_score is not None else scores,
        ],
        template=[
            ("T", hidden.dtype),
            ("K", intermediate),
            ("N", width),
            ("TOPK", top_k),
            ("SHARED", shared is not None),
            ("BITS", down.encoding.bits),
            ("GROUP", down.encoding.group_size),
            ("PACK", 64 // down.encoding.bits),
        ],
        grid=(64, width // 8, rows),
        threadgroup=(64, 1, 1),
        output_shapes=[hidden.shape],
        output_dtypes=[hidden.dtype],
    )[0]
