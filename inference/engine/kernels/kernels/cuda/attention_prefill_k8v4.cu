#define ATTENTION_AFFINE_HISTORY 1
#if SEISMIC_TUNE_HISTORY_STAGE
#define ATTENTION_HISTORY_STAGING 1
#endif
#define ATTENTION_KEY_CODE_WORDS SEISMIC_DIM_KC
#define ATTENTION_VALUE_CODE_WORDS SEISMIC_DIM_VC
#define ATTENTION_KEY_COEFFICIENTS SEISMIC_DIM_KP
#include "lib/attention/inputs.cuh"
#define ATTENTION_QUERY_GROUP SEISMIC_DIM_G
#define ATTENTION_INTERLEAVED SEISMIC_DIM_I
#define ATTENTION_SEPARATE SEISMIC_DIM_U
// attention_prefill_k8v4 (M >= 16): `attention_prefill` over affine K8/V4
// history (bodies in lib/attention/prefill.cuh): the prepare launch appends
// encoded rows; the attend launch's producer warps decode code tiles into F16
// operands beside its MMA warps (every product is F16).

#define ATTENTION_STAGES SEISMIC_TUNE_STAGES
#define ATTENTION_COLUMNS SEISMIC_TUNE_COLUMNS
#define ATTENTION_Q_REGISTERS SEISMIC_TUNE_QREG
#define ATTENTION_PRODUCER_WARPS SEISMIC_TUNE_PRODUCERS
// Two column warps over a 512-column head split the scores by key piece.
#define ATTENTION_SPLIT_SCORES 1
#include "lib/attention/prefill.cuh"

// Affine history planes: codes [T, KV, W * B / 32] u32 and group (scale,
// zero) pairs [T, KV, W / 16] f16, one aligned u32 per pair.
#define HISTORY()                                                                              \
    attention::AffineHistory {                                                                 \
        reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY_CODES)),      \
            reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY_COEFFICIENTS)), \
            reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE_CODES)),    \
            reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE_COEFFICIENTS)), \
            SEISMIC_HISTORY_KEY_CODES_STRIDE_0, SEISMIC_HISTORY_KEY_CODES_STRIDE_1,             \
            SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_0, SEISMIC_HISTORY_KEY_COEFFICIENTS_STRIDE_1, \
            SEISMIC_HISTORY_VALUE_CODES_STRIDE_0, SEISMIC_HISTORY_VALUE_CODES_STRIDE_1,         \
            SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_0,                                        \
            SEISMIC_HISTORY_VALUE_COEFFICIENTS_STRIDE_1, SEISMIC_PARAM_SLAB_ROWS                \
    }

extern "C" __global__ void __launch_bounds__(256)
    attention_prefill_k8v4_prepare(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::prepare(
        ATTENTION_INPUTS(), HISTORY(),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES)),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS)),
        reinterpret_cast<attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES)));
}

#define SPLIT()                                                                           \
    attention::prefill::Split {                                                           \
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS)),          \
            reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_STATISTICS)),    \
            reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_COUNTS)) \
    }

#if SEISMIC_TUNE_HISTORY_STAGE
#define STAGED_HISTORY() attention::ListedF16History { \
    reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_TILES)), \
    reinterpret_cast<const attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_HISTORY_VALID)), \
    reinterpret_cast<const attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_HISTORY_KEYS)), \
    reinterpret_cast<const attention::u16 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_HISTORY_VALUES)), \
    SEISMIC_DIM_HT, SEISMIC_HISTORY_TILES_STRIDE_1, SEISMIC_PARAM_SLAB_ROWS }

// One warp per32-row group; the exact union excludes unallocated/partial
// rows even when the caller's256-row list contains their containing page.
extern "C" __global__ void __launch_bounds__(32)
    attention_prefill_k8v4_history_union(SEISMIC_KERNEL_PARAMS) {
    const auto in = ATTENTION_INPUTS();
    const attention::u64 group = blockIdx.x;
    const int lane = threadIdx.x;
    const int page = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_TILES))[
        (group / 8) * SEISMIC_HISTORY_TILES_STRIDE_1];
    const long long first = static_cast<long long>(page) * 256 + (group % 8) * 32;
    const int count = page >= 0 && first >= 0 && static_cast<attention::u64>(first) < SEISMIC_DIM_T
        ? static_cast<int>(min(32ULL, SEISMIC_DIM_T - static_cast<attention::u64>(first))) : 0;
    const unsigned target = count == 32 ? 0xffffffffu : ((1u << count) - 1u);
    auto store = [&](unsigned mask) {
        if (lane == 0) reinterpret_cast<attention::u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_HISTORY_VALID))[group] = mask;
    };
    if (target == 0) { store(0); return; }
    const attention::u64 spans = SEISMIC_DIM_R;
    const attention::u64 intervals = SEISMIC_DIM_M * spans;
    if (intervals == 0) { store(0); return; }

    // Preserve the common complete-history route: one interval, one ballot.
    const auto initial = attention::span(in, 0, 0, spans);
    const long long token = first + lane;
    unsigned mask = __ballot_sync(0xffffffffu,
        lane < count && token >= initial.lo && token < initial.hi);
    if (mask == target) { store(mask); return; }

    // Each remaining interval is visited by one lane, rather than by every
    // visible-row lane. All clipping is signed/wide; length32 is explicit,
    // so neither a range mask nor its placement ever shifts by32.
    for (attention::u64 batch = 1; batch < intervals; batch += 32) {
        const attention::u64 index = batch + lane;
        unsigned bits = 0;
        if (index < intervals) {
            const auto interval = attention::span(in, index / max(1ULL, spans), index % max(1ULL, spans), spans);
            const long long lo = max(0LL, static_cast<long long>(interval.lo) - first);
            const long long hi = min(static_cast<long long>(count), static_cast<long long>(interval.hi) - first);
            if (hi > lo) {
                const int length = static_cast<int>(hi - lo);
                bits = length == 32 ? 0xffffffffu : (((1u << length) - 1u) << static_cast<int>(lo));
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            bits |= __shfl_xor_sync(0xffffffffu, bits, offset);
        mask |= bits;
        if (mask == target) break;
    }
    store(mask);
}

// Decode each unique logical row/head only once using the existing packed
// half decoder. Every scratch vector of a listed tile is initialized; an
// invalid union row never locates or reads an encoded slab. A list entry
// that names no tile is never looked up (`ListedF16History::listed`), so its
// scratch is left as it is: the launch's work follows the tiles listed.
extern "C" __global__ void __launch_bounds__(256)
    attention_prefill_k8v4_history_stage(SEISMIC_KERNEL_PARAMS) {
    const attention::u64 index = static_cast<attention::u64>(blockIdx.x) * blockDim.x + threadIdx.x;
    constexpr int chunks = attention::W / 8;
    if (index >= SEISMIC_DIM_HT * 256 * SEISMIC_DIM_KV * chunks) return;
    const int chunk = index % chunks;
    const int kv = (index / chunks) % SEISMIC_DIM_KV;
    const attention::u64 listed_row = index / (chunks * SEISMIC_DIM_KV);
    const attention::u64 slot = listed_row / 256;
    const int row = listed_row % 256;
    const auto staged = STAGED_HISTORY();
    if (staged.tiles[slot * staged.tile_stride] < 0) return;
    uint4 k = make_uint4(0, 0, 0, 0), v = k;
    if (staged.valid[slot * 8 + row / 32] & (1u << (row & 31))) {
        const int token = staged.tiles[slot * staged.tile_stride] * 256 + row;
        const auto source = HISTORY().vectors(token, kv);
        k = attention::prefill::key_chunk(source.key_codes[chunk * 2], source.key_codes[chunk * 2 + 1],
                                        source.key_pairs[chunk * 8 / attention::GROUP]);
        v = attention::prefill::value_chunk(source.value_codes[chunk], source.value_pairs[chunk * 8 / attention::GROUP]);
    }
    reinterpret_cast<uint4 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_HISTORY_KEYS))[index] = k;
    reinterpret_cast<uint4 *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_HISTORY_VALUES))[index] = v;
}
#define ATTEND_HISTORY() STAGED_HISTORY()
#else
#define ATTEND_HISTORY() HISTORY()
// Every declared launch names a kernel in every form. Without the staged
// history the two history launches are inactive (`when HISTORY_STAGE == 1`),
// so their kernels are empty here.
extern "C" __global__ void attention_prefill_k8v4_history_union(SEISMIC_KERNEL_PARAMS) {}
extern "C" __global__ void attention_prefill_k8v4_history_stage(SEISMIC_KERNEL_PARAMS) {}
#endif

extern "C" __global__ void __launch_bounds__(attention::prefill::MMA_THREADS + attention::prefill::PRODUCERS, 1)
    attention_prefill_k8v4_attend(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::attend(
        ATTENTION_INPUTS(), ATTEND_HISTORY(),
        attention::prefill::PrefillRows{SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_QUERIES),
                                      SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_KEYS),
                                      SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_VALUES),
                                      SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), SPLIT()},
        attention::prefill::Block{static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y),
                                static_cast<int>(blockIdx.z), static_cast<int>(gridDim.z)});
}

extern "C" __global__ void attention_prefill_k8v4_merge(SEISMIC_KERNEL_PARAMS) {
    attention::prefill::merge_partitions(ATTENTION_INPUTS(), SPLIT(), SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
}
