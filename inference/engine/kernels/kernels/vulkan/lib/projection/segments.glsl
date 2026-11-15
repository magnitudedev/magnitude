// Shared segment macros and zero-tile helper for Vulkan segmented projections.
#ifndef PROJECTION_ZEROED
#define PROJECTION_ZEROED(s) false
#endif
#ifndef PROJECTION_KIND0
#define PROJECTION_KIND0 PACKETS_W0
#endif
#ifndef PROJECTION_KIND1
#define PROJECTION_KIND1 PACKETS_W1
#endif
#ifndef PROJECTION_KIND2
#define PROJECTION_KIND2 PACKETS_W2
#endif
#ifndef PROJECTION_KIND3
#define PROJECTION_KIND3 PACKETS_W3
#endif
#define PROJECTION_PAIRED0 (SEISMIC_SEGMENT_WEIGHTS_0 == 2)
#if SEISMIC_SEGMENT_WEIGHTS_0 == 2 && !defined(PROJECTION_PARTNER0)
#error "segment 0 serves a pair: its entry names the partner (PROJECTION_PARTNER0)"
#elif SEISMIC_SEGMENT_WEIGHTS_0 != 2 && defined(PROJECTION_PARTNER0)
#error "segment 0 serves one weight: its entry names no partner"
#endif
#ifndef PROJECTION_PARTNER0
#define PROJECTION_PARTNER0 PROJECTION_WEIGHT0
#define PROJECTION_PARTNER_KIND0 PROJECTION_KIND0
#endif
#define PROJECTION_PAIRED1 (SEISMIC_SEGMENT_WEIGHTS_1 == 2)
#if SEISMIC_SEGMENT_WEIGHTS_1 == 2 && !defined(PROJECTION_PARTNER1)
#error "segment 1 serves a pair: its entry names the partner (PROJECTION_PARTNER1)"
#elif SEISMIC_SEGMENT_WEIGHTS_1 != 2 && defined(PROJECTION_PARTNER1)
#error "segment 1 serves one weight: its entry names no partner"
#endif
#ifndef PROJECTION_PARTNER1
#define PROJECTION_PARTNER1 PROJECTION_WEIGHT1
#define PROJECTION_PARTNER_KIND1 PROJECTION_KIND1
#endif
#define PROJECTION_PAIRED2 (SEISMIC_SEGMENT_WEIGHTS_2 == 2)
#if SEISMIC_SEGMENT_WEIGHTS_2 == 2 && !defined(PROJECTION_PARTNER2)
#error "segment 2 serves a pair: its entry names the partner (PROJECTION_PARTNER2)"
#elif SEISMIC_SEGMENT_WEIGHTS_2 != 2 && defined(PROJECTION_PARTNER2)
#error "segment 2 serves one weight: its entry names no partner"
#endif
#ifndef PROJECTION_PARTNER2
#define PROJECTION_PARTNER2 PROJECTION_WEIGHT2
#define PROJECTION_PARTNER_KIND2 PROJECTION_KIND2
#endif
#define PROJECTION_PAIRED3 (SEISMIC_SEGMENT_WEIGHTS_3 == 2)
#if SEISMIC_SEGMENT_WEIGHTS_3 == 2 && !defined(PROJECTION_PARTNER3)
#error "segment 3 serves a pair: its entry names the partner (PROJECTION_PARTNER3)"
#elif SEISMIC_SEGMENT_WEIGHTS_3 != 2 && defined(PROJECTION_PARTNER3)
#error "segment 3 serves one weight: its entry names no partner"
#endif
#ifndef PROJECTION_PARTNER3
#define PROJECTION_PARTNER3 PROJECTION_WEIGHT3
#define PROJECTION_PARTNER_KIND3 PROJECTION_KIND3
#endif

// X(s) for every segment s, in order.
#define PROJECTION_EACH_1(X) X(0)
#define PROJECTION_EACH_2(X) PROJECTION_EACH_1(X) X(1)
#define PROJECTION_EACH_3(X) PROJECTION_EACH_2(X) X(2)
#define PROJECTION_EACH_4(X) PROJECTION_EACH_3(X) X(3)
#define PROJECTION_EACH_OF(COUNT, X) PROJECTION_EACH_##COUNT(X)
#define PROJECTION_EACH_EXPANDED(COUNT, X) PROJECTION_EACH_OF(COUNT, X)
#define PROJECTION_EACH_SEGMENT(X) PROJECTION_EACH_EXPANDED(SEISMIC_PROJECTION_SEGMENTS, X)

// Zeros into `out_`'s rows row_start .. row_start + row_count and columns
// column_start .. column_start + column_count, within its rows x columns.
void projection_zero_tile(projection_epilogue out_, uint rows, uint columns, uint row_start, uint row_count,
    uint column_start, uint column_count) {
    for (uint item = gl_LocalInvocationIndex; item < row_count * column_count; item += gl_WorkGroupSize.x) {
        const uint row = row_start + item / column_count;
        const uint column = column_start + item % column_count;
        if (row < rows && column < columns)
            projection_put(out_, row, column, 0.0);
    }
}
