#include <seismic/slab.h>

// A floating plane is the plane's owner's (`SEISMIC_OUT_ROWS_PARAM`): its rows
// move as the stored elements, undecoded. Any other dense plane moves as bytes.
// Each thread moves one 16-byte chunk (`VEC` elements) along W.
kernel void copy_rows(
#ifdef SEISMIC_OUT_ROWS_PARAM
    SEISMIC_OUT_ROWS_PARAM(rows),
#else
    device const ulong *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
#endif
    device const uchar *from [[buffer(SEISMIC_BUFFER_FROM)]],
    device const uchar *to [[buffer(SEISMIC_BUFFER_TO)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
#if !defined(SEISMIC_ROWS_KIND_DENSE)
#error "copy_rows supports dense state planes only"
#endif
    constexpr uint elem_bytes = uint(ELEMENT_STRIDE_BYTES(SEISMIC_ROWS));
    constexpr uint VEC = elem_bytes >= 16u ? 1u : 16u / elem_bytes;
    const ulong chunks_per_head = (SEISMIC_DIM_W + ulong(VEC) - 1ul) / ulong(VEC);
    const ulong chunks_per_item = SEISMIC_DIM_KV * chunks_per_head;
    const ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_N * chunks_per_item) return;
    const ulong item = index / chunks_per_item;
    const uint rem = uint(index % chunks_per_item);
    const uint head = rem / uint(chunks_per_head);
    const uint column = (rem % uint(chunks_per_head)) * VEC;
    const uint count = min(VEC, uint(SEISMIC_DIM_W - ulong(column)));
    const int source_row = *reinterpret_cast<device const int *>(
        from + item * SEISMIC_FROM_STRIDE_0 * sizeof(int));
    const int destination_row = *reinterpret_cast<device const int *>(
        to + item * SEISMIC_TO_STRIDE_0 * sizeof(int));
    if (source_row < 0 || destination_row < 0
        || ulong(source_row) >= SEISMIC_DIM_T || ulong(destination_row) >= SEISMIC_DIM_T) return;
    const ulong slab_rows = ulong(SEISMIC_PARAM_SLAB_ROWS);
    const ulong source_offset = (ulong(source_row) % slab_rows) * SEISMIC_ROWS_STRIDE_0
        + ulong(head) * SEISMIC_ROWS_STRIDE_1 + ulong(column) * SEISMIC_ROWS_STRIDE_2;
    const ulong destination_offset = (ulong(destination_row) % slab_rows) * SEISMIC_ROWS_STRIDE_0
        + ulong(head) * SEISMIC_ROWS_STRIDE_1 + ulong(column) * SEISMIC_ROWS_STRIDE_2;
#ifdef SEISMIC_OUT_ROWS_PARAM
    const element::Out<ELEMENT_OF(SEISMIC_ROWS)> destination
        = slab::out_region(rows, ulong(destination_row) / slab_rows);
    const element::Out<ELEMENT_OF(SEISMIC_ROWS)> source
        = slab::out_region(rows, ulong(source_row) / slab_rows);
    if (SEISMIC_ROWS_STRIDE_2 == 1ul) {
        destination.move_stored(destination_offset, source, source_offset, ulong(count));
    } else {
        for (uint i = 0; i < count; ++i) {
            const ulong step = ulong(i) * SEISMIC_ROWS_STRIDE_2;
            destination.move_stored(destination_offset + step, source, source_offset + step, 1);
        }
    }
#else
    // An integer plane, of elements `elem_bytes` bytes each (its stride unit).
    device const uchar *source = slab::region<uchar>(rows, ulong(source_row) / slab_rows);
    device uchar *target = slab::region<uchar>(rows, ulong(destination_row) / slab_rows);
    if (SEISMIC_ROWS_STRIDE_2 == 1ul) {
        device uchar *dst = target + destination_offset * ulong(elem_bytes);
        device const uchar *src = source + source_offset * ulong(elem_bytes);
        const uint copy_bytes = count * elem_bytes;
        if (copy_bytes == 16u && ((ulong(dst) | ulong(src)) & 15ul) == 0ul) {
            *reinterpret_cast<device uint4 *>(dst) = *reinterpret_cast<device const uint4 *>(src);
        } else {
            for (uint byte = 0; byte < copy_bytes; ++byte)
                dst[byte] = src[byte];
        }
    } else {
        for (uint i = 0; i < count; ++i) {
            const ulong step = ulong(i) * SEISMIC_ROWS_STRIDE_2;
            for (uint byte = 0; byte < elem_bytes; ++byte)
                target[(destination_offset + step) * ulong(elem_bytes) + byte]
                    = source[(source_offset + step) * ulong(elem_bytes) + byte];
        }
    }
#endif
}
