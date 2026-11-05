// `thread_registers` and `register_file`: a kernel that keeps LIVE floats
// live at once, more than any thread's registers hold, so the compiler gives
// it as many as its workgroup size allows. Each value is the next step of
// one dependent chain from a single load, so none is cheaper to recompute
// than to keep, and each round reduces every value into one total before
// updating each by it, so all are live across the reduction (a stencil of
// neighbours would let the compiler sweep a narrow wavefront instead). It
// is formed, never run.
void main() {
    probe_f32 data = probe_f32(PROBE_ARGUMENT(0));
    const uint i = gl_GlobalInvocationID.x;
    float live[LIVE];
    live[0] = data.v[i % 4096u];
    [[unroll]] for (uint j = 1u; j < LIVE; ++j)
        live[j] = fma(live[j - 1u], live[j - 1u], float(j) * 0.001);
    [[unroll]] for (uint round = 0u; round < 4u; ++round) {
        float total = 0.0;
        [[unroll]] for (uint j = 0u; j < LIVE; ++j)
            total = fma(live[j], float(j + round), total);
        [[unroll]] for (uint j = 0u; j < LIVE; ++j)
            live[j] = fma(live[j], total, float(j));
    }
    float sum = 0.0;
    [[unroll]] for (uint j = 0u; j < LIVE; ++j)
        sum = fma(live[j], float(j + 1u), sum);
    data.v[4096u + i] = sum;
}
