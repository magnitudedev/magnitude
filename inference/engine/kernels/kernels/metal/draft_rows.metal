// draft_rows: normalize the successor embedding and conditioning in A, then
// project their joined [2D] row. Decode fuses these into one threadgroup per
// (32 outputs, row). Prefill joins each row once to scratch, then projects
// through the shared matrix library so a weight tile serves many rows.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_W0 SEISMIC_TABLE
#define KERNEL_W1 SEISMIC_COMBINE
#include "lib/projection/projection.h"
#include "lib/core/gathered.h"
#include "lib/core/reduce.h"

typedef activation::Act activation_element;
static_assert(activation_element::bytes == 2, "draft_rows requires a bf16 or f16 activation");
typedef ELEMENT_OF(SEISMIC_EMBEDDING_NORM) embedding_norm_element;
typedef ELEMENT_OF(SEISMIC_HIDDEN_NORM) hidden_norm_element;

constant constexpr uint draft_threads = 256;
constant constexpr uint draft_simdgroups = draft_threads / 32;
constant constexpr uint draft_rows_per_simdgroup = 4;
constant constexpr uint draft_outputs = draft_simdgroups * draft_rows_per_simdgroup;

#define DRAFT_ROWS_ARGUMENTS \
    device const int *tokens [[buffer(SEISMIC_BUFFER_TOKENS)]], \
    device const int *directory [[buffer(SEISMIC_BUFFER_DIRECTORY)]], \
    device const uchar *table [[buffer(SEISMIC_BUFFER_TABLE)]], \
    device const uchar *conditioning [[buffer(SEISMIC_BUFFER_CONDITIONING)]], \
    device const uchar *embedding_norm [[buffer(SEISMIC_BUFFER_EMBEDDING_NORM)]], \
    device const uchar *hidden_norm [[buffer(SEISMIC_BUFFER_HIDDEN_NORM)]], \
    device const uchar *combine [[buffer(SEISMIC_BUFFER_COMBINE)]], \
    SEISMIC_RESULT_0_PARAM(result_cells), \
    device uchar *joined_rows [[buffer(SEISMIC_BUFFER_SCRATCH_JOINED)]], \
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]

template <typename E>
struct DraftAct;

template <>
struct DraftAct<element::Bf16> {
    static inline float4 unpack4(ushort4 bits) {
        return as_type<float4>(uint4(bits) << 16);
    }
};

template <>
struct DraftAct<element::F16> {
    static inline float4 unpack4(half4 bits) {
        return float4(bits);
    }
};

inline float4 load4_tg(threadgroup const typename activation_element::storage *from) {
    return DraftAct<activation_element>::unpack4(
        vec<typename activation_element::storage, 4>(
            *reinterpret_cast<threadgroup const typename activation_element::packed4 *>(from)));
}

inline void store4_tg(threadgroup typename activation_element::storage *to, float4 values) {
    *reinterpret_cast<threadgroup typename activation_element::packed4 *>(to) =
        typename activation_element::packed4(activation_element::store4(values));
}

template <typename E>
inline float4 load4_strided(device const uchar *base, ulong offset, ulong stride) {
    device const typename E::storage *ptr = reinterpret_cast<device const typename E::storage *>(base);
    if (stride == 1ul)
        return E::load4(ptr + offset);
    return float4(
        E::load(ptr[offset]),
        E::load(ptr[offset + stride]),
        E::load(ptr[offset + 2ul * stride]),
        E::load(ptr[offset + 3ul * stride]));
}

inline void draft_join(
    device const int *tokens, device const int *directory, device const uchar *table,
    device const uchar *conditioning,
    device const uchar *embedding_norm, device const uchar *hidden_norm,
    constant ulong *seismic_words, threadgroup uchar *shared, ulong row,
    uint thread_index, uint sg, uint lane) {
    typedef typename activation_element::storage storage;
    const uint width = uint(SEISMIC_DIM_D);
    threadgroup storage *joined = reinterpret_cast<threadgroup storage *>(shared);
    threadgroup float *partials = reinterpret_cast<threadgroup float *>(shared + 4ul * width);
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    // The embedding row in A, packet by packet, and its sum of squares.
    projection::Weights<packets::W0> rows{{table, KERNEL_W0_LAYOUT(width), width}, nullptr};
    const long held = gathered::row(directory, SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R,
        SEISMIC_DIM_F, tokens[(row / SEISMIC_DIM_S) * SEISMIC_TOKENS_STRIDE_0]);
    const uint token = uint(held);
    float embedding_squares = 0.0f;
    // A row that names no table row embeds zeros.
    if (held < 0) {
        const uint4 zero8 = activation_element::pack8(float4(0.0f), float4(0.0f));
        threadgroup uint4 *joined4 = reinterpret_cast<threadgroup uint4 *>(joined);
        const uint vec8 = width / 8u;
        for (uint v = thread_index; v < vec8; v += draft_threads)
            joined4[v] = zero8;
        const storage zero = activation_element::store(0.0f);
        for (uint column = vec8 * 8u + thread_index; column < width; column += draft_threads)
            joined[column] = zero;
    } else {
        for (uint p = thread_index; p * 32u < width; p += draft_threads) {
            typename packets::W0::packet packet = rows.packet(token, p);
            for (uint step = 0; step < 4; ++step) {
                const uint base = 32u * p + 8u * step;
                if (base >= width)
                    break;
                float4 even, odd;
                packets::W0::exact_values(packet, step, even, odd);
                if (base + 8u <= width) {
                    const float4 e = float4(
                        activation_element::round(even.x), activation_element::round(even.y),
                        activation_element::round(even.z), activation_element::round(even.w));
                    const float4 o = float4(
                        activation_element::round(odd.x), activation_element::round(odd.y),
                        activation_element::round(odd.z), activation_element::round(odd.w));
                    *reinterpret_cast<threadgroup uint4 *>(joined + base) = activation_element::pack8(e, o);
                    embedding_squares = metal::fma(e.x, e.x, embedding_squares);
                    embedding_squares = metal::fma(o.x, o.x, embedding_squares);
                    embedding_squares = metal::fma(e.y, e.y, embedding_squares);
                    embedding_squares = metal::fma(o.y, o.y, embedding_squares);
                    embedding_squares = metal::fma(e.z, e.z, embedding_squares);
                    embedding_squares = metal::fma(o.z, o.z, embedding_squares);
                    embedding_squares = metal::fma(e.w, e.w, embedding_squares);
                    embedding_squares = metal::fma(o.w, o.w, embedding_squares);
                } else {
                    for (uint i = 0; i < 8u && base + i < width; ++i) {
                        const float value = activation_element::round((i & 1u) ? odd[i >> 1] : even[i >> 1]);
                        joined[base + i] = activation_element::store(value);
                        embedding_squares = metal::fma(value, value, embedding_squares);
                    }
                }
            }
        }
    }
    // The conditioning row and its sum of squares. When conditioning has the
    // same representation as A, cache its raw storage in joined[width + column]
    // so Pass 2 reads it from threadgroup memory instead of device memory.
    typedef ELEMENT_OF(SEISMIC_CONDITIONING) conditioning_element;
    const ulong source_row = row / SEISMIC_DIM_S;
    const ulong stream = row % SEISMIC_DIM_S;
    const ulong conditioning_row = source_row * SEISMIC_CONDITIONING_STRIDE_0
        + stream * width * SEISMIC_CONDITIONING_STRIDE_1;
    float hidden_squares = 0.0f;
#if ELEMENT_KIND(SEISMIC_CONDITIONING) == ELEMENT_KIND(SEISMIC_ELEMENT_A)
    device const storage *cond_storage = reinterpret_cast<device const storage *>(conditioning);
    if (SEISMIC_CONDITIONING_STRIDE_1 == 1ul) {
        device const storage *cond_base = cond_storage + conditioning_row;
        for (uint column = 4u * thread_index; column + 4u <= width; column += 4u * draft_threads) {
            const vec<storage, 4> raw = vec<storage, 4>(
                *reinterpret_cast<device const typename activation_element::packed4 *>(cond_base + column));
            *reinterpret_cast<threadgroup typename activation_element::packed4 *>(joined + width + column) =
                typename activation_element::packed4(raw);
            const float4 v = DraftAct<activation_element>::unpack4(raw);
            hidden_squares = metal::fma(v.x, v.x, hidden_squares);
            hidden_squares = metal::fma(v.y, v.y, hidden_squares);
            hidden_squares = metal::fma(v.z, v.z, hidden_squares);
            hidden_squares = metal::fma(v.w, v.w, hidden_squares);
        }
        for (uint column = (width & ~3u) + thread_index; column < width; column += draft_threads) {
            const storage raw = cond_base[column];
            joined[width + column] = raw;
            const float v = activation_element::load(raw);
            hidden_squares = metal::fma(v, v, hidden_squares);
        }
    } else {
        for (uint column = thread_index; column < width; column += draft_threads) {
            const storage raw = cond_storage[conditioning_row + ulong(column) * SEISMIC_CONDITIONING_STRIDE_1];
            joined[width + column] = raw;
            const float v = activation_element::load(raw);
            hidden_squares = metal::fma(v, v, hidden_squares);
        }
    }
#else
    device const typename conditioning_element::storage *cond_storage =
        reinterpret_cast<device const typename conditioning_element::storage *>(conditioning);
    if (SEISMIC_CONDITIONING_STRIDE_1 == 1ul) {
        device const typename conditioning_element::storage *cond_base = cond_storage + conditioning_row;
        for (uint column = 4u * thread_index; column + 4u <= width; column += 4u * draft_threads) {
            const float4 v = conditioning_element::load4(cond_base + column);
            hidden_squares = metal::fma(v.x, v.x, hidden_squares);
            hidden_squares = metal::fma(v.y, v.y, hidden_squares);
            hidden_squares = metal::fma(v.z, v.z, hidden_squares);
            hidden_squares = metal::fma(v.w, v.w, hidden_squares);
        }
        for (uint column = (width & ~3u) + thread_index; column < width; column += draft_threads) {
            const float v = conditioning_element::load(cond_base[column]);
            hidden_squares = metal::fma(v, v, hidden_squares);
        }
    } else {
        for (uint column = thread_index; column < width; column += draft_threads) {
            const float v = conditioning_element::load(
                cond_storage[conditioning_row + ulong(column) * SEISMIC_CONDITIONING_STRIDE_1]);
            hidden_squares = metal::fma(v, v, hidden_squares);
        }
    }
#endif
    // Fixed-order threadgroup sums (simdgroup sums, then the simdgroup
    // partials in index order); their barriers order every joined write
    // before these reads.
    float squares[2] = {embedding_squares, hidden_squares};
    reduce::group_sum_n<2, draft_simdgroups>(squares, partials, sg, lane);
    const float embedding_inverse = metal::rsqrt(squares[0] / float(width) + epsilon);
    const float hidden_inverse = metal::rsqrt(squares[1] / float(width) + epsilon);
    for (uint column = 4u * thread_index; column + 4u <= width; column += 4u * draft_threads) {
        const float4 emb_in = load4_tg(joined + column);
        const float4 emb_w = load4_strided<embedding_norm_element>(
            embedding_norm, ulong(column) * SEISMIC_EMBEDDING_NORM_STRIDE_0, SEISMIC_EMBEDDING_NORM_STRIDE_0);
        const float4 embedded = (emb_in * embedding_inverse) * emb_w;
#if ELEMENT_KIND(SEISMIC_CONDITIONING) == ELEMENT_KIND(SEISMIC_ELEMENT_A)
        const float4 hid_in = load4_tg(joined + width + column);
#else
        const float4 hid_in = load4_strided<conditioning_element>(
            conditioning, conditioning_row + ulong(column) * SEISMIC_CONDITIONING_STRIDE_1,
            SEISMIC_CONDITIONING_STRIDE_1);
#endif
        const float4 hid_w = load4_strided<hidden_norm_element>(
            hidden_norm, (stream * width + column) * SEISMIC_HIDDEN_NORM_STRIDE_0, SEISMIC_HIDDEN_NORM_STRIDE_0);
        const float4 hidden = (hid_in * hidden_inverse) * hid_w;
        store4_tg(joined + column, embedded);
        store4_tg(joined + width + column, hidden);
    }
    for (uint column = (width & ~3u) + thread_index; column < width; column += draft_threads) {
        const float embedded = activation_element::load(joined[column]) * embedding_inverse
            * element::at<embedding_norm_element>(embedding_norm, ulong(column) * SEISMIC_EMBEDDING_NORM_STRIDE_0);
#if ELEMENT_KIND(SEISMIC_CONDITIONING) == ELEMENT_KIND(SEISMIC_ELEMENT_A)
        const float hid_raw = activation_element::load(joined[width + column]);
#else
        const float hid_raw = element::at<conditioning_element>(conditioning,
            conditioning_row + ulong(column) * SEISMIC_CONDITIONING_STRIDE_1);
#endif
        const float hidden = hid_raw * hidden_inverse
            * element::at<hidden_norm_element>(hidden_norm, (stream * width + column) * SEISMIC_HIDDEN_NORM_STRIDE_0);
        joined[column] = activation_element::store(embedded);
        joined[width + column] = activation_element::store(hidden);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void draft_rows(DRAFT_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef typename activation_element::storage storage;
    const uint width = uint(SEISMIC_DIM_D);
    const ulong row = ulong(group.y);
    const element::Out<element::F32> result(result_cells);
    draft_join(tokens, directory, table, conditioning, embedding_norm, hidden_norm, seismic_words,
        shared, row, thread_index, sg, lane);
    const uint base_output = group.x * draft_outputs + sg * draft_rows_per_simdgroup;
    if (base_output >= width)
        return;
    const uint rows_active = metal::min(draft_rows_per_simdgroup, width - base_output);
    threadgroup const storage *joined = reinterpret_cast<threadgroup const storage *>(shared);
    // Four combine rows per simdgroup; lanes own packets of the 2D inputs,
    // loading each joined packet once from threadgroup memory and dotting it
    // against all four combine rows in registers.
    const uint inputs = 2u * width;
    projection::Weights<packets::W1> weights{{combine, KERNEL_W1_LAYOUT(inputs), inputs}, nullptr};
    typename projection::Weights<packets::W1>::located row_at[draft_rows_per_simdgroup];
    const bool together = weights.together<draft_rows_per_simdgroup>(width);
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < draft_rows_per_simdgroup; ++r) {
        if (r < rows_active) {
            if (r > 0 && together)
                row_at[r] = projection::Weights<packets::W1>::after(row_at[0], r);
            else
                row_at[r] = weights.locate(base_output + r);
        }
    }
    float sums[draft_rows_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint p = lane; p * 32u < inputs; p += 32u) {
        typename packets::W1::packet packet[draft_rows_per_simdgroup];
        _Pragma("clang loop unroll(full)")
        for (uint r = 0; r < draft_rows_per_simdgroup; ++r) {
            if (r < rows_active)
                packet[r] = weights.packet(row_at[r], p);
        }
        _Pragma("clang loop unroll(full)")
        for (uint step = 0; step < 4; ++step) {
            const uint base = 32u * p + 8u * step;
            if (base + 8u <= inputs) {
                const uint4 x_word = *reinterpret_cast<threadgroup const uint4 *>(joined + base);
                float4 xe, xo;
                activation_element::split8(x_word, xe, xo);
                _Pragma("clang loop unroll(full)")
                for (uint r = 0; r < draft_rows_per_simdgroup; ++r) {
                    if (r < rows_active) {
                        float4 we, wo;
                        packets::W1::values(packet[r], step, we, wo);
                        sums[r] = metal::fma(we.x, xe.x, sums[r]);
                        sums[r] = metal::fma(wo.x, xo.x, sums[r]);
                        sums[r] = metal::fma(we.y, xe.y, sums[r]);
                        sums[r] = metal::fma(wo.y, xo.y, sums[r]);
                        sums[r] = metal::fma(we.z, xe.z, sums[r]);
                        sums[r] = metal::fma(wo.z, xo.z, sums[r]);
                        sums[r] = metal::fma(we.w, xe.w, sums[r]);
                        sums[r] = metal::fma(wo.w, xo.w, sums[r]);
                    }
                }
            } else if (base < inputs) {
                float x[8];
                for (uint i = 0; i < 8u && base + i < inputs; ++i)
                    x[i] = activation_element::load(joined[base + i]);
                _Pragma("clang loop unroll(full)")
                for (uint r = 0; r < draft_rows_per_simdgroup; ++r) {
                    if (r < rows_active) {
                        float4 we, wo;
                        packets::W1::values(packet[r], step, we, wo);
                        for (uint i = 0; i < 8u && base + i < inputs; ++i)
                            sums[r] = metal::fma((i & 1u) ? wo[i >> 1] : we[i >> 1], x[i], sums[r]);
                    }
                }
            }
        }
    }
    const ulong row_base = (row / SEISMIC_DIM_S) * SEISMIC_RESULT_0_STRIDE_0
        + (row % SEISMIC_DIM_S) * width * SEISMIC_RESULT_0_STRIDE_1;
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < draft_rows_per_simdgroup; ++r) {
        if (r < rows_active) {
            const float sum = simd_sum(sums[r]);
            if (lane == 0)
                result.put(row_base + ulong(base_output + r) * SEISMIC_RESULT_0_STRIDE_1, sum);
        }
    }
}

kernel void draft_rows_join(DRAFT_ROWS_ARGUMENTS,
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef typename activation_element::storage storage;
    const uint width = uint(SEISMIC_DIM_D);
    const ulong row = ulong(group.y);
    draft_join(tokens, directory, table, conditioning, embedding_norm, hidden_norm, seismic_words,
        shared, row, thread_index, sg, lane);
    threadgroup const storage *joined = reinterpret_cast<threadgroup const storage *>(shared);
    device storage *out = reinterpret_cast<device storage *>(joined_rows) + row * 2ul * width;
    const uint total = 2u * width;
    if ((ulong(out) & 15ul) == 0ul) {
        const uint vecs = total / 8u;
        threadgroup const uint4 *src4 = reinterpret_cast<threadgroup const uint4 *>(joined);
        device uint4 *dst4 = reinterpret_cast<device uint4 *>(out);
        for (uint v = thread_index; v < vecs; v += draft_threads)
            dst4[v] = src4[v];
        for (uint column = vecs * 8u + thread_index; column < total; column += draft_threads)
            out[column] = joined[column];
    } else {
        for (uint column = thread_index; column < total; column += draft_threads)
            out[column] = joined[column];
    }
}

kernel void draft_rows_gemm(DRAFT_ROWS_ARGUMENTS,
    uint2 tile [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    PROJECTION_GEMM_SHARED(shared, 64, 64);
    const uint width = uint(SEISMIC_DIM_D);
    const uint inputs = 2u * width;
    projection::Plain<activation_element, projection::AllRows> in{joined_rows, inputs, 1, inputs, {}};
    projection::Store<element::F32> out{element::Out<element::F32>(result_cells),
        width, SEISMIC_RESULT_0_STRIDE_1, 0};
    projection::Weights<packets::W1> weights{{combine, KERNEL_W1_LAYOUT(inputs), inputs}, nullptr};
    projection::gemm<packets::W1, 64, 64>(in, out, weights, uint(SEISMIC_DIM_M * SEISMIC_DIM_S),
        width, inputs, tile.y, tile.x, shared, sg, lane);
}
