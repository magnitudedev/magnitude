// P3/R6 streaming-read bandwidth probe (CUDA).
// Each lane issues UNROLL independent 16 B `ld.global.nc.v4.u32` loads per step of a
// grid-stride loop, adds them into a register accumulator, and each warp writes one
// __reduce_add_sync. The sum of all outputs equals the 32-bit wrapping sum of every
// word in the buffer, which is checked for every configuration: each byte is read
// exactly once and no read can be eliminated.
//
//   nvcc -O3 -arch=sm_121 stream_read.cu -o stream-read
//   ./stream-read [--sizes-gb 1,2,4] [--repetitions 10] > bandwidth-<host>.json
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                                   \
    do {                                                                              \
        cudaError_t status = (call);                                                  \
        if (status != cudaSuccess) {                                                  \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(status)); \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

__global__ void fill(uint32_t* data, uint64_t words) {
    uint64_t id = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < words) data[id] = uint32_t(id) * 0x9E3779B1u;
}

__device__ __forceinline__ uint4 load_nc(const uint4* address) {
    uint4 v;
    asm("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(address));
    return v;
}

template <int UNROLL>
__global__ void stream_read(const uint4* __restrict__ src, uint32_t* __restrict__ out, uint32_t count) {
    uint32_t grid = gridDim.x * blockDim.x;
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    uint32_t i = id;
    for (; i + (UNROLL - 1) * grid < count; i += UNROLL * grid) {
        uint4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = load_nc(src + i + u * grid);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            acc.x += v[u].x; acc.y += v[u].y; acc.z += v[u].z; acc.w += v[u].w;
        }
    }
    for (; i < count; i += grid) {
        uint4 v = load_nc(src + i);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    uint32_t total = __reduce_add_sync(0xffffffffu, acc.x + acc.y + acc.z + acc.w);
    if ((threadIdx.x & 31) == 0) out[id / 32] = total;
}

struct Kernel {
    int unroll;
    void (*function)(const uint4*, uint32_t*, uint32_t);
};
static const Kernel kernels[] = {{1, stream_read<1>}, {2, stream_read<2>}, {4, stream_read<4>}, {8, stream_read<8>}};

static std::vector<int> parse_ints(const char* text) {
    std::vector<int> values;
    for (const char* p = text; *p;) {
        values.push_back(atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return values;
}
static std::vector<double> parse_doubles(const char* text) {
    std::vector<double> values;
    for (const char* p = text; *p;) {
        values.push_back(atof(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return values;
}
static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}
static std::string join(const std::vector<int>& values) {
    std::string s;
    for (size_t i = 0; i < values.size(); ++i) s += (i ? ", " : "") + std::to_string(values[i]);
    return s;
}

struct Result {
    uint64_t buffer_bytes;
    int unroll, threads_per_block;
    std::string grid;
    int blocks;
    uint64_t grid_threads, resident_threads, bytes_in_flight;
    int resident_blocks_per_sm;
    std::vector<double> seconds;
    double median_seconds, gb_per_second;
};

static void print_result(const Result& r, const char* indent) {
    printf("%s{\"bufferBytes\": %llu, \"unroll\": %d, \"threadsPerBlock\": %d, \"grid\": \"%s\", \"blocks\": %d, "
           "\"gridThreads\": %llu, \"residentBlocksPerSM\": %d, \"residentThreads\": %llu, \"bytesInFlight\": %llu, "
           "\"medianSeconds\": %.9g, \"gbPerSecond\": %.6g, \"checksumCorrect\": true, \"gpuSeconds\": [",
           indent, (unsigned long long)r.buffer_bytes, r.unroll, r.threads_per_block, r.grid.c_str(), r.blocks,
           (unsigned long long)r.grid_threads, r.resident_blocks_per_sm, (unsigned long long)r.resident_threads,
           (unsigned long long)r.bytes_in_flight, r.median_seconds, r.gb_per_second);
    for (size_t i = 0; i < r.seconds.size(); ++i) printf("%s%.9g", i ? ", " : "", r.seconds[i]);
    printf("]}");
}

int main(int argc, char** argv) {
    std::vector<int> sizes_gb = {1, 2, 4};
    std::vector<int> unrolls = {1, 2, 4, 8};
    std::vector<int> threads_per_block = {64, 128, 256, 512, 1024};
    std::vector<double> grid_multipliers = {0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128};
    int repetitions = 10, warmup = 2;
    for (int a = 1; a < argc; a += 2) {
        if (a + 1 >= argc) { fprintf(stderr, "%s requires a value\n", argv[a]); return 1; }
        if (!strcmp(argv[a], "--sizes-gb")) sizes_gb = parse_ints(argv[a + 1]);
        else if (!strcmp(argv[a], "--unrolls")) unrolls = parse_ints(argv[a + 1]);
        else if (!strcmp(argv[a], "--threads-per-block")) threads_per_block = parse_ints(argv[a + 1]);
        else if (!strcmp(argv[a], "--grid-multipliers")) grid_multipliers = parse_doubles(argv[a + 1]);
        else if (!strcmp(argv[a], "--repetitions")) repetitions = atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "--warmup")) warmup = atoi(argv[a + 1]);
        else { fprintf(stderr, "unknown flag %s\n", argv[a]); return 1; }
    }
    if (repetitions < 10) { fprintf(stderr, "at least 10 timed repetitions are required\n"); return 1; }

    char host[256];
    gethostname(host, sizeof host);
    int device_index;
    CHECK(cudaGetDevice(&device_index));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device_index));
    int runtime_version, driver_version;
    CHECK(cudaRuntimeGetVersion(&runtime_version));
    CHECK(cudaDriverGetVersion(&driver_version));
    int sms = prop.multiProcessorCount;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    std::vector<Result> results;
    std::vector<size_t> best_per_size;
    for (int size_gb : sizes_gb) {
        uint64_t bytes = uint64_t(size_gb) << 30;
        uint64_t words = bytes / 4;
        uint32_t vectors = uint32_t(bytes / 16);
        if (uint64_t(vectors) * 9 >= 0xffffffffull) { fprintf(stderr, "32-bit loop indices require a smaller buffer\n"); return 1; }
        uint32_t* source;
        CHECK(cudaMalloc(&source, bytes));
        fill<<<unsigned((words + 255) / 256), 256>>>(source, words);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        // Sum over w of w * 0x9E3779B1 (mod 2^32) = 0x9E3779B1 * n(n-1)/2 (mod 2^32).
        uint32_t expected = uint32_t((words * (words - 1) / 2) * 0x9E3779B1ull);
        size_t first = results.size();
        for (int unroll : unrolls) {
            const Kernel* kernel = nullptr;
            for (const Kernel& k : kernels) if (k.unroll == unroll) kernel = &k;
            if (!kernel) { fprintf(stderr, "unsupported unroll %d\n", unroll); return 1; }
            for (int tpb : threads_per_block) {
                int per_sm;
                CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel->function, tpb, 0));
                int full = int((uint64_t(vectors) + uint64_t(unroll) * tpb - 1) / (uint64_t(unroll) * tpb));
                std::vector<std::pair<std::string, int>> grids;
                for (double m : grid_multipliers) {
                    int blocks = std::max(1, int(m * sms));
                    char label[32];
                    snprintf(label, sizeof label, "%gxSMs", m);
                    if (blocks < full) grids.push_back({label, blocks});
                }
                grids.push_back({"full", full});
                for (auto& [label, blocks] : grids) {
                    uint64_t grid_threads = uint64_t(blocks) * tpb;
                    uint32_t* out;
                    CHECK(cudaMalloc(&out, grid_threads / 32 * 4));
                    Result r{bytes, unroll, tpb, label, blocks, grid_threads, 0, 0, per_sm, {}, 0, 0};
                    r.resident_threads = std::min<uint64_t>(grid_threads, uint64_t(per_sm) * sms * tpb);
                    r.bytes_in_flight = r.resident_threads * unroll * 16;
                    for (int sample = 0; sample < warmup + repetitions; ++sample) {
                        CHECK(cudaEventRecord(start));
                        kernel->function<<<blocks, tpb>>>(reinterpret_cast<const uint4*>(source), out, vectors);
                        CHECK(cudaEventRecord(stop));
                        CHECK(cudaEventSynchronize(stop));
                        CHECK(cudaGetLastError());
                        float ms;
                        CHECK(cudaEventElapsedTime(&ms, start, stop));
                        if (sample >= warmup) r.seconds.push_back(ms / 1e3);
                    }
                    std::vector<uint32_t> sums(grid_threads / 32);
                    CHECK(cudaMemcpy(sums.data(), out, sums.size() * 4, cudaMemcpyDeviceToHost));
                    CHECK(cudaFree(out));
                    uint32_t total = 0;
                    for (uint32_t s : sums) total += s;
                    if (total != expected) {
                        fprintf(stderr, "checksum mismatch: %u != %u (%d GB unroll %d tpb %d %s)\n", total, expected, size_gb, unroll, tpb, label.c_str());
                        return 1;
                    }
                    r.median_seconds = median(r.seconds);
                    r.gb_per_second = double(bytes) / r.median_seconds / 1e9;
                    fprintf(stderr, "%d GB unroll %d tpb %d %s: %.1f GB/s\n", size_gb, unroll, tpb, label.c_str(), r.gb_per_second);
                    results.push_back(r);
                }
            }
        }
        size_t best = first;
        for (size_t i = first; i < results.size(); ++i) if (results[i].gb_per_second > results[best].gb_per_second) best = i;
        best_per_size.push_back(best);
        CHECK(cudaFree(source));
    }
    size_t best = best_per_size[0];
    for (size_t i : best_per_size) if (results[i].gb_per_second > results[best].gb_per_second) best = i;

    printf("{\n  \"probe\": \"stream_read\",\n  \"backend\": \"cuda\",\n  \"host\": \"%s\",\n  \"device\": \"%s\",\n", host, prop.name);
    printf("  \"computeCapability\": \"%d.%d\",\n  \"multiprocessors\": %d,\n  \"cudaRuntime\": %d,\n  \"cudaDriver\": %d,\n",
           prop.major, prop.minor, sms, runtime_version, driver_version);
    printf("  \"allocation\": \"cudaMalloc\",\n  \"load\": \"ld.global.nc.v4.u32\",\n");
    printf("  \"parameters\": {\"sizesGB\": [%s], \"unrolls\": [%s], \"threadsPerBlock\": [%s], \"gridMultipliers\": [",
           join(sizes_gb).c_str(), join(unrolls).c_str(), join(threads_per_block).c_str());
    for (size_t i = 0; i < grid_multipliers.size(); ++i) printf("%s%g", i ? ", " : "", grid_multipliers[i]);
    printf("], \"repetitions\": %d, \"warmup\": %d},\n", repetitions, warmup);
    printf("  \"definitions\": {\n"
           "    \"gbPerSecond\": \"bufferBytes / median(cudaEventElapsedTime around one launch) / 1e9\",\n"
           "    \"bytesInFlight\": \"residentThreads * unroll * 16: independent 16 B loads outstanding at once over the resident threads\",\n"
           "    \"residentThreads\": \"min(gridThreads, cudaOccupancyMaxActiveBlocksPerMultiprocessor * multiprocessors * threadsPerBlock)\",\n"
           "    \"grid\": \"NxSMs = max(1, floor(N * multiprocessors)) blocks (grid-stride loop); full = one step per lane covering the buffer\",\n"
           "    \"checksumCorrect\": \"wrapping sum of all warp outputs equals the wrapping sum of every 32-bit word\"\n  },\n");
    printf("  \"bestPerSize\": [\n");
    for (size_t i = 0; i < best_per_size.size(); ++i) {
        print_result(results[best_per_size[i]], "    ");
        printf(i + 1 < best_per_size.size() ? ",\n" : "\n");
    }
    printf("  ],\n  \"best\": ");
    print_result(results[best], "");
    printf(",\n  \"results\": [\n");
    for (size_t i = 0; i < results.size(); ++i) {
        print_result(results[i], "    ");
        printf(i + 1 < results.size() ? ",\n" : "\n");
    }
    printf("  ]\n}\n");
    return 0;
}
