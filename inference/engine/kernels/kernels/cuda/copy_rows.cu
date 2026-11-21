// copy_rows: rows[to[item], head, :] = rows[from[item], head, :] for dense state
// planes of one element type; one thread per 16-byte chunk along W. Rows outside
// plane (including -1) are skipped.
#if !defined(SEISMIC_ROWS_KIND_DENSE)
#error "copy_rows supports dense state planes only"
#endif
#include <seismic/element.cuh>
#include <seismic/slab.cuh>

extern "C" __global__ void copy_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    constexpr u64 unit = ELEMENT_STRIDE_BYTES(SEISMIC_ROWS);
    constexpr u64 CHUNK = unit >= 16 ? 1 : (unit > 0 ? 16 / unit : 16);
    const u64 chunks_per_head = ((u64)SEISMIC_DIM_W + CHUNK - 1) / CHUNK;
    const u64 chunks_per_item = (u64)SEISMIC_DIM_KV * chunks_per_head;
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (u64)SEISMIC_DIM_N * chunks_per_item)
        return;
    const u64 item = index / chunks_per_item;
    const u64 rem = index % chunks_per_item;
    const u64 head = rem / chunks_per_head;
    const u64 column = (rem % chunks_per_head) * CHUNK;
    const u64 remaining = (u64)SEISMIC_DIM_W - column;
    const u64 count = remaining < CHUNK ? remaining : CHUNK;
    const int *from = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FROM));
    const int *to = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TO));
    const int source_row = from[item * SEISMIC_FROM_STRIDE_0];
    const int destination_row = to[item * SEISMIC_TO_STRIDE_0];
    if (source_row < 0 || destination_row < 0 || (u64)source_row >= SEISMIC_DIM_T
        || (u64)destination_row >= SEISMIC_DIM_T)
        return;
    const u64 at = head * (u64)SEISMIC_ROWS_STRIDE_1 + column * (u64)SEISMIC_ROWS_STRIDE_2;
#if defined(SEISMIC_OUT_ROWS)
    // A floating plane: its owner-written elements move as stored.
    const auto rows = SEISMIC_OUT_ROWS;
    const auto dst_row = rows.row((u64)destination_row, SEISMIC_PARAM_SLAB_ROWS, SEISMIC_ROWS_STRIDE_0);
    const auto src_row = rows.view_row((u64)source_row, SEISMIC_PARAM_SLAB_ROWS, SEISMIC_ROWS_STRIDE_0);
    if (SEISMIC_ROWS_STRIDE_2 == 1) {
        dst_row.move_stored(at, src_row.offset(at), count);
    } else {
        for (u64 i = 0; i < count; ++i) {
            const u64 step = i * (u64)SEISMIC_ROWS_STRIDE_2;
            dst_row.move_stored(at + step, src_row.offset(at + step));
        }
    }
#else
    // An integer plane, of elements `unit` bytes each (its stride unit).
    const unsigned int *table = reinterpret_cast<const unsigned int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROWS));
    const unsigned char *src_row = slab::row(table, (u64)source_row, SEISMIC_PARAM_SLAB_ROWS,
        (u64)SEISMIC_ROWS_STRIDE_0 * unit);
    unsigned char *dst_row = slab::row(table, (u64)destination_row, SEISMIC_PARAM_SLAB_ROWS,
        (u64)SEISMIC_ROWS_STRIDE_0 * unit);
    if (SEISMIC_ROWS_STRIDE_2 == 1) {
        const unsigned char *src = src_row + at * unit;
        unsigned char *dst = dst_row + at * unit;
        const u64 byte_count = count * unit;
        const u64 align_bits = reinterpret_cast<u64>(dst) | reinterpret_cast<u64>(src);
        u64 offset = 0;
        if ((align_bits & 15u) == 0 && byte_count >= 16) {
            for (; offset + 16 <= byte_count; offset += 16) {
                *reinterpret_cast<uint4 *>(dst + offset) = *reinterpret_cast<const uint4 *>(src + offset);
            }
        }
        if ((align_bits & 7u) == 0) {
            for (; offset + 8 <= byte_count; offset += 8) {
                *reinterpret_cast<u64 *>(dst + offset) = *reinterpret_cast<const u64 *>(src + offset);
            }
        }
        if ((align_bits & 3u) == 0) {
            for (; offset + 4 <= byte_count; offset += 4) {
                *reinterpret_cast<unsigned int *>(dst + offset) =
                    *reinterpret_cast<const unsigned int *>(src + offset);
            }
        }
        for (; offset < byte_count; ++offset) {
            dst[offset] = src[offset];
        }
    } else {
        for (u64 i = 0; i < count; ++i) {
            const u64 elem_at = at + i * (u64)SEISMIC_ROWS_STRIDE_2;
            for (u64 byte = 0; byte < unit; ++byte) {
                dst_row[elem_at * unit + byte] = src_row[elem_at * unit + byte];
            }
        }
    }
#endif
}
