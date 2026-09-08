uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint ci = sg % NC, hi = sg / NC;
uint block = threadgroup_position_in_grid.y;
uint item = threadgroup_position_in_grid.z;
uint qt = item % QGROUPS, ht = (item / QGROUPS) % HEAD_GROUPS;
uint kh = (item / (QGROUPS * HEAD_GROUPS)) % HK;
uint row = item / (QGROUPS * HEAD_GROUPS * HK);
uint local_head = hi * HP, head = ht * HG + local_head;
constexpr int PK = DK / 32, PV = DV / 32;
uint pos = positions[row];
uint origin = WINDOW > 0 && pos + 1 > WINDOW ? pos + 1 - WINDOW : 0;
// Partition logical visibility, independent of page-table capacity padding.
uint visible = pos + TQ - origin;
uint subspan = (visible + BLOCKS * NC - 1) / (BLOCKS * NC);
uint span = subspan * NC;
uint begin = origin + block * span + ci * subspan;
uint end = min(origin + (block + 1) * span, begin + subspan);
end = min(end, pos + min(uint(TQ), (qt + 1) * QT));
// Fixed tile loops must unroll: dynamically indexed arrays spill registers.
// Keep cached inputs in their exact storage dtype; widen at FP32 arithmetic use.
In query[HP][QT][PK];
float output[HP][QT][PV];
float maxima[HP][QT], sums[HP][QT];
_Pragma("clang loop unroll(full)") for (uint h = 0; h < HP; ++h)
_Pragma("clang loop unroll(full)") for (uint t = 0; t < QT; ++t) {
    uint token = qt * QT + t;
    _Pragma("clang loop unroll(full)") for (uint i = 0; i < PK; ++i)
        query[h][t][i] = token < TQ
            ? q[((row * HQ + kh * G + head + h) * TQ + token) * DK + lane * PK + i]
            : In(0);
    _Pragma("clang loop unroll(full)") for (uint i = 0; i < PV; ++i) output[h][t][i] = 0.0f;
    maxima[h][t] = -INFINITY; sums[h][t] = 0.0f;
}
uint p = begin;
while (p < end) {
    bool tail = TAIL > 0 && p >= uint(starts[row]);
    uint stop = tail ? end : min(end, (p / PAGE + 1) * PAGE);
    if (TAIL > 0 && !tail) stop = min(stop, uint(starts[row]));
    size_t address = tail ? (size_t(row) * HK + kh) * TAIL + p - starts[row]
        : size_t(kh) * CAPACITY + uint(pages[row * TABLE + p / PAGE]) * PAGE + p % PAGE;
    auto kp = tail ? tk : k;
    auto vp = tail ? tv : v;
    // Issue a bounded tile of independent loads before dependent scores.
    // Consumption stays in logical key order, including page/tail edges.
    for (; p < stop; p += KT, address += KT) {
        In key[KT][PK], value[KT][PV];
        _Pragma("clang loop unroll(full)") for (uint u = 0; u < KT; ++u) {
            if (p + u >= stop) continue;
            if (PK % 4 == 0) {
                auto src = reinterpret_cast<const device vec<In, 4>*>(
                    kp + (address + u) * DK + lane * PK);
                _Pragma("clang loop unroll(full)") for (uint i = 0; i < PK / 4; ++i) {
                    vec<In, 4> x = src[i];
                    _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) key[u][i * 4 + j] = x[j];
                }
            } else _Pragma("clang loop unroll(full)") for (uint i = 0; i < PK; ++i)
                key[u][i] = kp[(address + u) * DK + lane * PK + i];
            if (PV % 4 == 0) {
                auto src = reinterpret_cast<const device vec<In, 4>*>(
                    vp + (address + u) * DV + lane * PV);
                _Pragma("clang loop unroll(full)") for (uint i = 0; i < PV / 4; ++i) {
                    vec<In, 4> x = src[i];
                    _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) value[u][i * 4 + j] = x[j];
                }
            } else _Pragma("clang loop unroll(full)") for (uint i = 0; i < PV; ++i)
                value[u][i] = vp[(address + u) * DV + lane * PV + i];
        }
        _Pragma("clang loop unroll(full)") for (uint u = 0; u < KT; ++u) {
            if (p + u >= stop) continue;
            _Pragma("clang loop unroll(full)") for (uint h = 0; h < HP; ++h)
            _Pragma("clang loop unroll(full)") for (uint t = 0; t < QT; ++t) {
                uint token = qt * QT + t;
                if (token >= TQ || p + u > pos + token
                    || (WINDOW > 0 && p + u + WINDOW <= pos + token)) continue;
                float score = 0.0f;
                _Pragma("clang loop unroll(full)") for (uint i = 0; i < PK; ++i)
                    score += float(query[h][t][i]) * float(key[u][i]);
                score = simd_sum(score) * scale[0];
                float peak = max(maxima[h][t], score);
                float a = exp(maxima[h][t] - peak), b = exp(score - peak);
                sums[h][t] = sums[h][t] * a + b;
                _Pragma("clang loop unroll(full)") for (uint i = 0; i < PV; ++i)
                    output[h][t][i] = output[h][t][i] * a + b * float(value[u][i]);
                maxima[h][t] = peak;
            }
        }
        // Do not skip the first keys after an unaligned page boundary.
        if (stop - p < KT) { p = stop; break; }
    }
}
threadgroup float scratch[NC > 1 ? HG * NC * (DV + 2) : 1];
_Pragma("clang loop unroll(full)") for (uint t = 0; t < QT; ++t) {
    if (NC > 1) {
        _Pragma("clang loop unroll(full)") for (uint h = 0; h < HP; ++h) {
            uint at = ((local_head + h) * NC + ci) * (DV + 2);
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < PV; ++i)
                scratch[at + lane * PV + i] = output[h][t][i];
            if (lane == 0) { scratch[at + DV] =
                maxima[h][t]; scratch[at + DV + 1] = sums[h][t]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (ci == 0 && qt * QT + t < TQ) _Pragma("clang loop unroll(full)") for (uint h = 0; h < HP; ++h) {
        float peak = maxima[h][t], denom = sums[h][t];
        float result[PV];
        _Pragma("clang loop unroll(full)") for (uint i = 0; i < PV; ++i) result[i] = output[h][t][i];
        if (NC > 1) {
            uint at = (local_head + h) * NC * (DV + 2);
            peak = -INFINITY;
            for (uint c = 0; c < NC; ++c) peak = max(peak, scratch[at + c * (DV + 2) + DV]);
            denom = 0.0f;
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < PV; ++i) result[i] = 0.0f;
            for (uint c = 0; c < NC; ++c) {
                uint src = at + c * (DV + 2);
                float factor =
                    scratch[src + DV + 1] > 0 ? exp(scratch[src + DV] - peak) : 0.0f;
                denom += factor * scratch[src + DV + 1];
                for (uint i =
                    0; i < PV; ++i) result[i] += factor * scratch[src + lane * PV + i];
            }
        }
        size_t out =
            ((size_t(row) * HQ + kh * G + head + h) * TQ + qt * QT + t) * BLOCKS + block;
        _Pragma("clang loop unroll(full)") for (uint i = 0; i < PV; ++i)
            partial[out * DV + lane * PV + i] =
                Out(BLOCKS == 1 ? result[i] / denom : result[i]);
        if (lane == 0) { maximum[out] = peak; denominator[out] = denom; }
    }
    if (NC > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
}
