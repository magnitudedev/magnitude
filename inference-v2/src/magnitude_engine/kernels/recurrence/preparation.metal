constexpr int TG = DK;                       // one q/k head per threadgroup
uint c   = thread_position_in_grid.x;         // channel in [0, CK)
uint tid = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;    // batch row
const uint P = CK + CV + 2 * HV;
const device T* st = state + row * (K - 1) * CK;
device T* ns = new_state + row * (K - 1) * CK;
threadgroup float part[TG / 32];
// window of the last K-1 raw inputs for this channel (starts as the carried state)
float win[K - 1];
for (int t = 0; t < K - 1; ++t) win[t] = static_cast<float>(st[t * CK + c]);
for (int tt = 0; tt < TT; ++tt) {
    const device T* pr = proj + (row * TT + tt) * P;
    float xin = static_cast<float>(pr[c]);
    float acc = 0.0f;
    for (int t = 0; t < K - 1; ++t) acc += static_cast<float>(w[c * K + t]) * win[t];
    acc += static_cast<float>(w[c * K + (K - 1)]) * xin;
    float r = static_cast<float>(static_cast<T>(acc));
    float val = float(T(r * float(magnitude_native_sigmoid<T, false>(r))));
    for (int t = 0; t < K - 2; ++t) win[t] = win[t + 1];
    win[K - 2] = xin;
    float ss = simd_sum(val * val);
    if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int s = 0; s < TG / 32; ++s) tot += part[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool is_q = c < KEYDIM;
    bool is_k = (c >= KEYDIM) && (c < 2 * KEYDIM);
    float out = val;
    if (is_q || is_k) {
        float inv = metal::rsqrt(tot / float(TG) + 1e-6f);
        float normed = static_cast<float>(static_cast<T>(val * inv));
        // MLX converts scalar factors to the activation dtype before multiplying.
        float scale = is_q ? (1.0f / float(DK))
            : float(T(metal::rsqrt(float(DK))));
        out = static_cast<float>(static_cast<T>(normed * scale));
    }
    y[(row * TT + tt) * CK + c] = static_cast<T>(out);
    if (c >= 2 * KEYDIM) {
        uint zi = c - 2 * KEYDIM;
        z[(row * TT + tt) * CV + zi] = pr[CK + zi];
    }
    if (c < HV) {
        float bb = static_cast<float>(pr[CK + CV + c]);
        beta[(row * TT + tt) * HV + c] = magnitude_native_sigmoid<T, true>(bb);
        g[(row * TT + tt) * HV + c] = magnitude_decay<T>(pr[CK + CV + HV + c], dt_bias[c], A_log[c]);
    }
}
for (int t = 0; t < K - 1; ++t) ns[t * CK + c] = static_cast<T>(win[t]);
