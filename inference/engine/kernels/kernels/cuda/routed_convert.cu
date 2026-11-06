// Expert streaming, a scratch round's stored records (`routed_convert`
// in routed.seismic); the CUDA form of `metal/routed_convert.metal`:
// when the list's sequence is this step's, entry i below its count converts
// record `from` into scratch slot `to` by windows
// (lib/streaming/window.cuh), `repack_weight`'s conversion, or a dense
// record by rows, `import_dense`'s: `from` is a conveyor record, below V
// (any view of stored-form records), or, negative, scratch slot -1 - from,
// which holds the stored bytes the host read there (`listed_source`). It is
// the body of one block of its round's written grid (`routed_scratch_grids`: CONVERT_TILE_ROWS-row
// tiles by the round's records), and blocks stride over the listed records'
// row tiles (a dense record's rows), so a launch costs what its list holds.

#include "lib/streaming/control.cuh"
#include "lib/streaming/window.cuh"
#include "../streaming/list.hpp"

REPACK_BINDING(ConveyorBinding, SEISMIC_CONVERSION_ST_U, SEISMIC_CONVEYOR, SEISMIC_DESTINATION, SEISMIC_DIM_N, SEISMIC_DIM_K)

// The grid's tile is the declaration's: the host writes ceil(N /
// CONVERT_TILE_ROWS) blocks a record.
static_assert(ConveyorBinding::dense() || ConveyorBinding::tile_rows() == SEISMIC_CONST_CONVERT_TILE_ROWS,
              "a convert block's tile is repack's");

using element::u64;
using element::u8;

// The block's list words (streaming/list.hpp), in its `Shared`.
struct ConvertWords {
    int group[STREAMING_LIST_WORDS];
};

#ifdef SEISMIC_FORMING_ROUTED_CONVERT
__device__ void routed_convert_block(SEISMIC_KERNEL_PARAMS) {
    const int *list = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LIST));
    const int *step = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    __shared__ streaming::Shared<ConvertWords, ConveyorBinding> shared;
    const u64 stride = SEISMIC_LIST_STRIDE_0;
    const u64 entries =
        streaming_step_entries(list, stride, step[0], shared.words.group, threadIdx.x);
    const u64 per_record = streaming::record_items<ConveyorBinding>(seismic_words_value);
    const u8 *conveyor = SEISMIC_PTR(SEISMIC_BUFFER_CONVEYOR);
    u8 *destination = SEISMIC_PTR(SEISMIC_BUFFER_DESTINATION);
    const u64 first = blockIdx.x + (u64)gridDim.x * blockIdx.y, step_items = (u64)gridDim.x * gridDim.y;
    if constexpr (ConveyorBinding::dense()) {
        STREAMING_LIST_ITEMS(at, entries, per_record, first, step_items) {
            streaming_list_pair(list, stride, at, per_record, entries, shared.words.group, threadIdx.x, blockDim.x);
            const int from = streaming_list_from(shared.words.group), to = streaming_list_to(shared.words.group);
            __syncthreads();
            streaming::convert_row<ConveyorBinding>(
                streaming::listed_source<ConveyorBinding>(conveyor, destination, seismic_words_value, from), destination,
                seismic_words_value, 0, (u64)to, at % per_record);
            __syncthreads();
        }
    } else {
        // Row tiles, each window loaded while the one before converts.
        const auto pair = [&](u64 at) {
            streaming_list_pair(list, stride, at, per_record, entries, shared.words.group, threadIdx.x, blockDim.x);
            return int2{streaming_list_from(shared.words.group), streaming_list_to(shared.words.group)};
        };
        streaming::convert_records<ConveyorBinding>(conveyor, destination, seismic_words_value, entries * per_record,
                                                    first, step_items, per_record, pair, shared);
    }
}
#endif
