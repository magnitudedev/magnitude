// P2: GB10 `mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}.{bf16,f16}.f32` determinism,
// swap-AB equivalence, and accumulation-order probe (inline PTX, no mma headers).
//
// One kernel computes D = P * Q^T for row-major, K-contiguous P (rowsP x K) and
// Q (rowsQ x K): each warp owns one 16 x 8 output tile and chains one mma per k16 step
// (acc = mma(a_k, b_k, acc), acc starting at zero). For activations X (M x K) and weights
// W (N x K):
//   standard: P = X, Q = W  -> out[m][n]  (activations are the A operand)
//   swapped:  P = W, Q = X  -> out[n][m]  (weights are the A operand, M rows sit in N = 8;
//                                          the K1 CUDA GEMV form)
// Reports (a) bitwise determinism over repeated launches with different block shapes,
// tile orders and concurrent split launches, (b) bitwise standard-vs-swapped agreement,
// (c) agreement with host models of the accumulation (sequential f32 FMA, and exact
// per-k16 sum added to the accumulator with round-to-nearest or truncation), and
// (d) crafted single-dot-product cases that separate those models.
//
//   nvcc -O3 -arch=sm_121 mma_determinism.cu -o mma-determinism
//   ./mma-determinism [--n 1024] [--k 1024] [--repetitions 10] > mma-determinism-<host>.json
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // host-side f16 conversions only

// The exact per-k16 reference needs every partial sum to be exact: IEEE binary128
// long double (aarch64 Linux, i.e. the Grace host of GB10).
static_assert(LDBL_MANT_DIG == 113, "the exact block reference requires binary128 long double");

#define CHECK(call)                                                                   \
    do {                                                                              \
        cudaError_t status = (call);                                                  \
        if (status != cudaSuccess) {                                                  \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(status)); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

enum class Type { bf16, f16 };
static const char* type_name(Type t) { return t == Type::bf16 ? "bf16" : "f16"; }

// ---------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------
template <Type T>
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    if constexpr (T == Type::bf16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
}

// Two consecutive 16-bit elements (k, k+1) of row r, or zero past the last row.
__device__ __forceinline__ uint32_t pair(const uint16_t* m, int rows, int r, int K, int k) {
    return r < rows ? *reinterpret_cast<const uint32_t*>(m + size_t(r) * K + k) : 0u;
}

template <Type T>
__global__ void tile_mma(const uint16_t* __restrict__ P, int rowsP, const uint16_t* __restrict__ Q, int rowsQ, int K,
                         float* __restrict__ out, int tileOffset, int tileCount, int reversed) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int linear = blockIdx.x * (blockDim.x / 32) + warp;
    if (linear >= tileCount) return;
    int tile = reversed ? tileOffset + tileCount - 1 - linear : tileOffset + linear;
    int tilesQ = (rowsQ + 7) / 8;
    int p0 = (tile / tilesQ) * 16, q0 = (tile % tilesQ) * 8;
    int g = lane / 4, t = lane % 4;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < K; k0 += 16) {
        // m16n8k16 16-bit fragments (PTX ISA "Matrix Fragments for mma.m16n8k16"):
        // A: {row g, k 2t..2t+1}, {row g+8, k 2t..}, {row g, k 2t+8..}, {row g+8, k 2t+8..}
        // B: {k 2t..2t+1, col g}, {k 2t+8..2t+9, col g}
        uint32_t a[4] = {pair(P, rowsP, p0 + g, K, k0 + 2 * t), pair(P, rowsP, p0 + g + 8, K, k0 + 2 * t),
                         pair(P, rowsP, p0 + g, K, k0 + 2 * t + 8), pair(P, rowsP, p0 + g + 8, K, k0 + 2 * t + 8)};
        uint32_t b[2] = {pair(Q, rowsQ, q0 + g, K, k0 + 2 * t), pair(Q, rowsQ, q0 + g, K, k0 + 2 * t + 8)};
        mma<T>(c, a, b);
    }
    // D: c0, c1 = row g, cols 2t, 2t+1; c2, c3 = row g+8, cols 2t, 2t+1.
    int rows[2] = {p0 + g, p0 + g + 8};
    for (int h = 0; h < 2; ++h)
        for (int j = 0; j < 2; ++j) {
            int r = rows[h], col = q0 + 2 * t + j;
            if (r < rowsP && col < rowsQ) out[size_t(r) * rowsQ + col] = c[2 * h + j];
        }
}

// ---------------------------------------------------------------------------------
// Host numerics
// ---------------------------------------------------------------------------------
static float bf16_to_float(uint16_t v) { uint32_t b = uint32_t(v) << 16; float f; memcpy(&f, &b, 4); return f; }
static uint16_t float_to_bf16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    return uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
static float f16_to_float(uint16_t v) { __half_raw r; r.x = v; return __half2float(__half(r)); }
static uint16_t float_to_f16(float f) { return __half_raw(__float2half_rn(f)).x; }
static float decode(Type t, uint16_t v) { return t == Type::bf16 ? bf16_to_float(v) : f16_to_float(v); }
static uint16_t encode(Type t, float f) {
    uint16_t v = t == Type::bf16 ? float_to_bf16(f) : float_to_f16(f);
    if (decode(t, v) != f) { fprintf(stderr, "%g is not exact in %s\n", f, type_name(t)); exit(1); }
    return v;
}
static uint32_t bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static int64_t ordered(float f) { uint32_t b = bits(f); return b & 0x80000000u ? -int64_t(b & 0x7fffffffu) : int64_t(b); }
static uint64_t ulp_distance(float a, float b) { int64_t d = ordered(a) - ordered(b); return uint64_t(d < 0 ? -d : d); }
static float round_nearest(long double v) { return float(v); }
static float round_toward_zero(long double v) {
    float f = float(v);
    if (fabsl((long double)f) > fabsl(v)) f = nextafterf(f, 0.0f);
    return f;
}

struct Models { float sequential_fma, block_nearest, block_truncate; };
static Models reference(const float* p, const float* q, int K) {
    Models m{0.f, 0.f, 0.f};
    for (int k = 0; k < K; ++k) m.sequential_fma = fmaf(p[k], q[k], m.sequential_fma);
    for (int k0 = 0; k0 < K; k0 += 16) {
        long double s_near = m.block_nearest, s_trunc = m.block_truncate;
        long double block = 0;
        for (int k = k0; k < k0 + 16; ++k) block += (long double)p[k] * (long double)q[k];
        m.block_nearest = round_nearest(s_near + block);
        m.block_truncate = round_toward_zero(s_trunc + block);
    }
    return m;
}

// Deterministic generator (splitmix64).
static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t next_u64() {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// "uniform": uniform in [-1, 1] rounded to the type; "wide": random sign, exponent in
// [-12, 12] and random mantissa, so products span 2^-24 .. 2^26 and alignment matters.
static uint16_t random_value(Type t, bool wide) {
    uint64_t r = next_u64();
    float f;
    if (!wide) f = float(double(r >> 11) / double(1ull << 53) * 2.0 - 1.0);
    else f = ((r & 1) ? -1.f : 1.f) * ldexpf(1.f + float((r >> 8) & 0xffff) / 65536.f, int((r >> 32) % 25) - 12);
    return t == Type::bf16 ? float_to_bf16(f) : float_to_f16(f);
}

// ---------------------------------------------------------------------------------
// Launch variants
// ---------------------------------------------------------------------------------
struct Variant { const char* name; int warps_per_block; int reversed; int split; };
static const Variant variants[] = {
    {"1warp-forward", 1, 0, 0},
    {"4warps-forward", 4, 0, 0},
    {"4warps-reversed", 4, 1, 0},
    {"8warps-split-concurrent", 8, 1, 1},
};

template <Type T>
static void launch(const Variant& v, const uint16_t* P, int rowsP, const uint16_t* Q, int rowsQ, int K, float* out) {
    int tiles = ((rowsP + 15) / 16) * ((rowsQ + 7) / 8);
    int threads = 32 * v.warps_per_block;
    if (!v.split) {
        tile_mma<T><<<(tiles + v.warps_per_block - 1) / v.warps_per_block, threads>>>(P, rowsP, Q, rowsQ, K, out, 0, tiles, v.reversed);
        CHECK(cudaGetLastError());
    } else {
        // Two halves on two non-blocking streams, the second half issued first.
        cudaStream_t s[2];
        CHECK(cudaStreamCreateWithFlags(&s[0], cudaStreamNonBlocking));
        CHECK(cudaStreamCreateWithFlags(&s[1], cudaStreamNonBlocking));
        int half = tiles / 2;
        tile_mma<T><<<(tiles - half + v.warps_per_block - 1) / v.warps_per_block, threads, 0, s[1]>>>(P, rowsP, Q, rowsQ, K, out, half, tiles - half, v.reversed);
        CHECK(cudaGetLastError());
        if (half > 0) {
            tile_mma<T><<<(half + v.warps_per_block - 1) / v.warps_per_block, threads, 0, s[0]>>>(P, rowsP, Q, rowsQ, K, out, 0, half, v.reversed);
            CHECK(cudaGetLastError());
        }
        CHECK(cudaStreamSynchronize(s[0]));
        CHECK(cudaStreamSynchronize(s[1]));
        CHECK(cudaStreamDestroy(s[0]));
        CHECK(cudaStreamDestroy(s[1]));
    }
    CHECK(cudaDeviceSynchronize());
}

template <Type T>
static std::vector<float> run(const Variant& v, const std::vector<uint16_t>& P, int rowsP, const std::vector<uint16_t>& Q, int rowsQ, int K) {
    uint16_t *dP, *dQ; float* dOut;
    CHECK(cudaMalloc(&dP, P.size() * 2));
    CHECK(cudaMalloc(&dQ, Q.size() * 2));
    CHECK(cudaMalloc(&dOut, size_t(rowsP) * rowsQ * 4));
    CHECK(cudaMemcpy(dP, P.data(), P.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * 2, cudaMemcpyHostToDevice));
    // NaN fill so an unwritten element can never match.
    CHECK(cudaMemset(dOut, 0xff, size_t(rowsP) * rowsQ * 4));
    launch<T>(v, dP, rowsP, dQ, rowsQ, K, dOut);
    std::vector<float> out(size_t(rowsP) * rowsQ);
    CHECK(cudaMemcpy(out.data(), dOut, out.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dP)); CHECK(cudaFree(dQ)); CHECK(cudaFree(dOut));
    return out;
}

struct Agreement { uint64_t compared = 0, unequal = 0, max_ulp = 0; double max_abs = 0; };
static void accumulate(Agreement& a, float actual, float expected) {
    a.compared++;
    if (bits(actual) != bits(expected)) a.unequal++;
    if (std::isnan(actual) || std::isnan(expected)) { a.max_ulp = UINT64_MAX; return; }
    a.max_ulp = std::max(a.max_ulp, ulp_distance(actual, expected));
    a.max_abs = std::max(a.max_abs, double(fabsf(actual - expected)));
}
static std::string json(const Agreement& a) {
    char s[256];
    snprintf(s, sizeof s, "{\"compared\": %llu, \"unequalBits\": %llu, \"maxUlp\": %llu, \"maxAbs\": %.9g}",
             (unsigned long long)a.compared, (unsigned long long)a.unequal, (unsigned long long)a.max_ulp, a.max_abs);
    return s;
}

template <Type T>
static std::string problem(bool wide, int M, int N, int K, int repetitions) {
    std::vector<uint16_t> X(size_t(M) * K), W(size_t(N) * K);
    for (auto& v : X) v = random_value(T, wide);
    for (auto& v : W) v = random_value(T, wide);

    std::vector<float> standard = run<T>(variants[0], X, M, W, N, K);   // [m][n]
    std::vector<float> swapped = run<T>(variants[0], W, N, X, M, K);    // [n][m]

    // (a) determinism of each form against its own first run.
    std::string determinism = "[";
    for (int form = 0; form < 2; ++form) {
        for (size_t vi = 0; vi < sizeof variants / sizeof variants[0]; ++vi) {
            Agreement a;
            for (int r = 0; r < repetitions; ++r) {
                std::vector<float> again = form == 0 ? run<T>(variants[vi], X, M, W, N, K) : run<T>(variants[vi], W, N, X, M, K);
                const std::vector<float>& base = form == 0 ? standard : swapped;
                for (size_t i = 0; i < base.size(); ++i) accumulate(a, again[i], base[i]);
            }
            char head[128];
            snprintf(head, sizeof head, "%s{\"form\": \"%s\", \"variant\": \"%s\", \"runs\": %d, \"agreement\": ",
                     determinism.size() > 1 ? ", " : "", form == 0 ? "standard" : "swapped", variants[vi].name, repetitions);
            determinism += std::string(head) + json(a) + "}";
        }
    }
    determinism += "]";

    // (b) swapped vs standard, (c) host models.
    Agreement swap, seq_std, near_std, trunc_std, seq_swp, near_swp, trunc_swp;
    std::vector<float> x(K), w(K);
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) x[k] = decode(T, X[size_t(m) * K + k]);
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) w[k] = decode(T, W[size_t(n) * K + k]);
            float s = standard[size_t(m) * N + n], t = swapped[size_t(n) * M + m];
            accumulate(swap, t, s);
            // The standard form multiplies x_k * w_k, the swapped form w_k * x_k.
            Models ms = reference(x.data(), w.data(), K), mt = reference(w.data(), x.data(), K);
            accumulate(seq_std, s, ms.sequential_fma);
            accumulate(near_std, s, ms.block_nearest);
            accumulate(trunc_std, s, ms.block_truncate);
            accumulate(seq_swp, t, mt.sequential_fma);
            accumulate(near_swp, t, mt.block_nearest);
            accumulate(trunc_swp, t, mt.block_truncate);
        }
    }
    char head[256];
    snprintf(head, sizeof head, "{\"type\": \"%s\", \"distribution\": \"%s\", \"M\": %d, \"N\": %d, \"K\": %d, ",
             type_name(T), wide ? "wide" : "uniform", M, N, K);
    std::string s = std::string(head) + "\"determinism\": " + determinism + ", \"swappedVsStandard\": " + json(swap) +
        ", \"standardVsModels\": {\"sequentialFma\": " + json(seq_std) + ", \"blockExactRoundNearest\": " + json(near_std) +
        ", \"blockExactTruncate\": " + json(trunc_std) + "}, \"swappedVsModels\": {\"sequentialFma\": " + json(seq_swp) +
        ", \"blockExactRoundNearest\": " + json(near_swp) + ", \"blockExactTruncate\": " + json(trunc_swp) + "}}";
    fprintf(stderr, "%s %s M=%d: swap unequal %llu, seqFMA unequal %llu, blockRN unequal %llu, blockRZ unequal %llu\n",
            type_name(T), wide ? "wide" : "uniform", M, (unsigned long long)swap.unequal, (unsigned long long)seq_std.unequal,
            (unsigned long long)near_std.unequal, (unsigned long long)trunc_std.unequal);
    return s;
}

// (d) crafted single dot products: products p_k = x_k * w_k chosen so the models differ.
struct Crafted { const char* name; int K; std::vector<std::pair<int, std::pair<float, float>>> terms; };
static std::vector<Crafted> crafted_cases() {
    const float big = 4096.f;  // big * big = 2^24, exact in bf16 and f16
    std::vector<Crafted> cases;
    auto ones = [](int from, int to, int skip) {
        std::vector<std::pair<int, std::pair<float, float>>> t;
        for (int k = from; k < to; ++k) if (k != skip) t.push_back({k, {1.f, 1.f}});
        return t;
    };
    { Crafted c{"2^24 at k0, +1 at k1..15", 16, ones(0, 16, 0)}; c.terms.push_back({0, {big, big}}); cases.push_back(c); }
    { Crafted c{"+1 at k0..14, 2^24 at k15", 16, ones(0, 16, 15)}; c.terms.push_back({15, {big, big}}); cases.push_back(c); }
    { Crafted c{"2^24 at k8, +1 elsewhere", 16, ones(0, 16, 8)}; c.terms.push_back({8, {big, big}}); cases.push_back(c); }
    { Crafted c{"2^24 at k7, +1 elsewhere", 16, ones(0, 16, 7)}; c.terms.push_back({7, {big, big}}); cases.push_back(c); }
    cases.push_back({"2^24, +1, -2^24 at k0,1,2", 16, {{0, {big, big}}, {1, {1.f, 1.f}}, {2, {-big, big}}}});
    cases.push_back({"+1, 2^24, -2^24 at k0,1,2", 16, {{0, {1.f, 1.f}}, {1, {big, big}}, {2, {-big, big}}}});
    cases.push_back({"2^24 at k0, -2^24 at k15, +1 at k8", 16, {{0, {big, big}}, {15, {-big, big}}, {8, {1.f, 1.f}}}});
    { Crafted c{"k-tile 0: 2^24; k-tile 1: sixteen +1", 32, ones(16, 32, -1)}; c.terms.push_back({0, {big, big}}); cases.push_back(c); }
    { Crafted c{"k-tile 0: 2^24; k-tile 1: sixteen +1/16", 32, {}}; c.terms.push_back({0, {big, big}});
      for (int k = 16; k < 32; ++k) c.terms.push_back({k, {0.25f, 0.25f}}); cases.push_back(c); }
    { Crafted c{"k-tile 0: 2^24; k-tile 1: 3/2 as 1 + 1/2", 32, {{0, {big, big}}, {16, {1.f, 1.f}}, {17, {0.5f, 1.f}}}}; cases.push_back(c); }
    return cases;
}

template <Type T>
static std::string crafted() {
    std::string s = "[";
    for (const Crafted& c : crafted_cases()) {
        std::vector<uint16_t> X(c.K, encode(T, 0.f)), W(c.K, encode(T, 0.f));
        for (auto& [k, xw] : c.terms) { X[k] = encode(T, xw.first); W[k] = encode(T, xw.second); }
        float standard = run<T>(variants[0], X, 1, W, 1, c.K)[0];
        float swapped = run<T>(variants[0], W, 1, X, 1, c.K)[0];
        std::vector<float> x(c.K), w(c.K);
        for (int k = 0; k < c.K; ++k) { x[k] = decode(T, X[k]); w[k] = decode(T, W[k]); }
        Models m = reference(x.data(), w.data(), c.K);
        long double exact = 0;
        for (int k = 0; k < c.K; ++k) exact += (long double)x[k] * w[k];
        char line[512];
        snprintf(line, sizeof line,
                 "%s{\"case\": \"%s\", \"exact\": %.17Lg, \"standard\": %.9g, \"swapped\": %.9g, \"sequentialFma\": %.9g, "
                 "\"blockExactRoundNearest\": %.9g, \"blockExactTruncate\": %.9g}",
                 s.size() > 1 ? ", " : "", c.name, exact, standard, swapped, m.sequential_fma, m.block_nearest, m.block_truncate);
        s += line;
    }
    return s + "]";
}

int main(int argc, char** argv) {
    int N = 1024, K = 1024, repetitions = 10;
    for (int a = 1; a < argc; a += 2) {
        if (a + 1 >= argc) { fprintf(stderr, "%s requires a value\n", argv[a]); return 1; }
        if (!strcmp(argv[a], "--n")) N = atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "--k")) K = atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "--repetitions")) repetitions = atoi(argv[a + 1]);
        else { fprintf(stderr, "unknown flag %s\n", argv[a]); return 1; }
    }
    if (K % 16 != 0) { fprintf(stderr, "--k must be a multiple of 16\n"); return 1; }

    char host[256];
    gethostname(host, sizeof host);
    cudaDeviceProp prop;
    int device_index;
    CHECK(cudaGetDevice(&device_index));
    CHECK(cudaGetDeviceProperties(&prop, device_index));

    std::string problems = "[";
    const int ms[] = {1, 5, 8, 16};
    for (int wide = 0; wide < 2; ++wide)
        for (int M : ms) {
            if (problems.size() > 1) problems += ",\n    ";
            problems += problem<Type::bf16>(wide, M, N, K, repetitions);
            problems += ",\n    " + problem<Type::f16>(wide, M, N, K, repetitions);
        }
    problems += "]";

    printf("{\n  \"probe\": \"mma_determinism\",\n  \"backend\": \"cuda\",\n  \"host\": \"%s\",\n  \"device\": \"%s\",\n", host, prop.name);
    printf("  \"computeCapability\": \"%d.%d\",\n", prop.major, prop.minor);
    printf("  \"instructions\": [\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\", \"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\"],\n");
    printf("  \"parameters\": {\"N\": %d, \"K\": %d, \"repetitionsPerVariant\": %d},\n", N, K, repetitions);
    printf("  \"definitions\": {\n"
           "    \"standard\": \"D[m][n] = X * W^T with X (activations) as the A operand and W as B\",\n"
           "    \"swapped\": \"D[n][m] = W * X^T with W (weights) as the A operand and the M activation rows in N = 8\",\n"
           "    \"determinism\": \"bitwise comparison of every repetition of a launch variant with the first 1warp-forward run of the same form\",\n"
           "    \"sequentialFma\": \"host f32 y = fmaf(p_k, q_k, y) for k ascending, in the form's operand order\",\n"
           "    \"blockExactRoundNearest\": \"per k16 step: acc = round_nearest(acc + exact sum of the 16 products) in binary128\",\n"
           "    \"blockExactTruncate\": \"per k16 step: acc = round_toward_zero(acc + exact sum of the 16 products) in binary128\",\n"
           "    \"crafted\": \"single dot products (M = N = 1) whose products are chosen so the models disagree\"\n  },\n");
    printf("  \"problems\": %s,\n", problems.c_str());
    printf("  \"crafted\": {\"bf16\": %s,\n    \"f16\": %s}\n}\n", crafted<Type::bf16>().c_str(), crafted<Type::f16>().c_str());
    return 0;
}
