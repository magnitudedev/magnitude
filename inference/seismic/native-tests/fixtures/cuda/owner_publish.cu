// Group r writes result row r through the owner's handle with operation r;
// each thread writes spans of eight elements of the row. The result is
// canonical (contiguous rows of N, N % 8 == 0), so a span's first element is
// aligned to every vector store the handle makes. A value moves from `x` to
// the handle as loaded, without arithmetic, so a NaN payload reaches the
// owner unchanged.
#include <seismic/element.cuh>

typedef ELEMENT_OF(SEISMIC_RESULT_0) PublishElement;

extern "C" __global__ void owner_publish(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    const element::Out<PublishElement> out = SEISMIC_RESULT_0_OUT;
    const unsigned long long row = blockIdx.x;
    for (unsigned long long first = 8ull * threadIdx.x; first < SEISMIC_DIM_N; first += 8ull * blockDim.x) {
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = x[row * SEISMIC_X_STRIDE_0 + (first + i) * SEISMIC_X_STRIDE_1];
        }
        const unsigned long long at = row * SEISMIC_RESULT_0_STRIDE_0 + first;
        switch (row) {
        case 0: // put
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                out.put(at + i, v[i]);
            }
            break;
        case 1: // put2 at even indices: the vector (F32) and packed-convert (16-bit) stores
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                out.put2(at + i, v[i], v[i + 1]);
            }
            break;
        case 2: // put2 at odd indices: two scalar stores each
            out.put(at, v[0]);
#pragma unroll
            for (int i = 1; i < 7; i += 2) {
                out.put2(at + i, v[i], v[i + 1]);
            }
            out.put(at + 7, v[7]);
            break;
        case 3: // put_span<2>
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const float pair[2] = {v[i], v[i + 1]};
                out.put_span<2>(at + i, pair);
            }
            break;
        case 4: // put_span<4>
#pragma unroll
            for (int i = 0; i < 8; i += 4) {
                const float four[4] = {v[i], v[i + 1], v[i + 2], v[i + 3]};
                out.put_span<4>(at + i, four);
            }
            break;
        case 5: // put_span<8>
            out.put_span<8>(at, v);
            break;
        default: // put_span<8> through a sub-view
            out.offset(at).put_span<8>(0, v);
            break;
        }
    }
}
