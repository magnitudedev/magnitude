// The hyper-connection streams' Metal helpers (streams.seismic): a row's
// S streams collapsed into one by per-stream weights, shared by
// `stream_collapse` and `stream_mix_sinkhorn`.
//
// This file is independent of any entry ABI: a caller passes its views'
// strides.

namespace streams {

// out[c] = sum_s carried[s] x[s, c] in stream order, for the row's columns
// c = first, first + step, .. below h: x's stream s, column c at
// s * stream_stride + c * column_stride, out column c at c * out_stride.
template <uint S>
inline void collapse(device const float *x, ulong stream_stride, ulong column_stride, thread const float *carried,
    device float *out, ulong out_stride, uint h, uint first, uint step) {
    for (uint column = first; column < h; column += step) {
        float sum = 0.0f;
        for (uint stream = 0; stream < S; ++stream)
            sum += carried[stream] * x[ulong(stream) * stream_stride + ulong(column) * column_stride];
        out[ulong(column) * out_stride] = sum;
    }
}

// As above, by the weights `weights` holds, weight s at s * weight_stride.
template <uint S>
inline void collapse(device const float *x, ulong stream_stride, ulong column_stride, device const float *weights,
    ulong weight_stride, device float *out, ulong out_stride, uint h, uint first, uint step) {
    float carried[S];
    for (uint stream = 0; stream < S; ++stream)
        carried[stream] = weights[ulong(stream) * weight_stride];
    collapse<S>(x, stream_stride, column_stride, carried, out, out_stride, h, first, step);
}

} // namespace streams
