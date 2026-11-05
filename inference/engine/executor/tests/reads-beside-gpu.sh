#!/usr/bin/env bash
# A prompt's expert reads into a reserved bank (magnitude-executor's
# a_prompts_reads_land_in_a_reserved_bank_at_the_heaps_rate) alone, beside a
# served prompt's host work (a_prompts_reads_beside_a_served_prompts_host_work),
# and beside a memory-bound GPU copy, a compute-bound GPU loop, and one
# thread spinning on host-mapped words as a parked device's pulse wait does
# (two acquire.sys loads a spin, a third every 64: streaming/park.hpp), each
# alone and all three at once; each burner first alone.
# Prepare: cargo test --release -p magnitude-executor --lib --no-run
#
# A burner's life is bounded three ways: it stops launching once its stop
# file appears or its own deadline passes, and synchronises before it frees
# anything; and each kernel ends at its own budget on the device clock
# (%globaltimer), whatever the host does, so a killed process leaves no
# kernel spinning.
set -euo pipefail
export MAGNITUDE_ARTIFACTS="$MRIG_ARTIFACTS"
BIN=$("$MRIG_PYTHON" "$MRIG_RIG" magnitude_rig.host resolve --list "$MRIG_EXECUTABLES" lib magnitude_executor)
"$MRIG_PYTHON" "$MRIG_RIG" magnitude_rig.host inputs "$MRIG_ARTIFACTS"
echo "binary: $BIN"
TESTS="a_prompts_reads_land_in_a_reserved_bank_at_the_heaps_rate a_prompts_reads_beside_a_served_prompts_host_work"
for test in $TESTS; do
  "$BIN" --list --ignored | grep -qx "experts::source::tests::$test: test" || { echo "no test $test in $BIN"; exit 1; }
  echo "runs: $BIN --ignored --nocapture --exact experts::source::tests::$test"
done
python3 -c 'import ctypes; ctypes.CDLL("libcuda.so.1")' || { echo "no libcuda.so.1"; exit 1; }
ls /usr/local/cuda*/lib64/libnvrtc.so* > /dev/null || { echo "no libnvrtc"; exit 1; }
BURN=$(mktemp --suffix .py)
cat > "$BURN" <<'EOF'
import ctypes, glob, os, sys, time
kind, seconds, stop = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cuda = ctypes.CDLL("libcuda.so.1")
P, S, U, Q = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint64
cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(Q), S]
cuda.cuMemHostAlloc.argtypes = [ctypes.POINTER(P), S, U]
cuda.cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(Q), P, U]
cuda.cuLaunchKernel.argtypes = [P, U, U, U, U, U, U, U, P, ctypes.POINTER(P), P]
paths = glob.glob(os.environ.get("SEISMIC_NVRTC_DIRECTORY", "/nonexistent") + "/**/libnvrtc.so*", recursive=True) + glob.glob("/usr/local/cuda*/**/libnvrtc.so*", recursive=True)
nvrtc = ctypes.CDLL(sorted(paths)[0])
def ok(code, what):
    if code != 0: raise SystemExit(f"{what}: {code}")
# Every kernel ends once `budget` ns of the device clock have passed since it began.
src = rb'''
__device__ unsigned long long now() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
extern "C" __global__ void copy(const float4* a, float4* b, unsigned long long n, unsigned long long budget) {
  const unsigned long long began = now();
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) {
    b[i] = a[i];
    if ((i & 0xffff) == 0 && now() - began > budget) return;
  }
}
extern "C" __global__ void spin(float* o, int iters, unsigned long long budget) {
  const unsigned long long began = now();
  float x = threadIdx.x, y = 1.0001f, z = 0.5f, w = 0.25f;
  for (int i = 0; i < iters; i++) {
    x = x * y + z; y = y * w + x; z = z * x + w; w = w * z + y;
    if ((i & 0x3ff) == 0 && now() - began > budget) break;
  }
  if (x + y + z + w == 12345.f) o[0] = x;
}
extern "C" __global__ void pulse(const unsigned long long* w, unsigned int spins, unsigned long long* o, unsigned long long budget) {
  const unsigned long long began = now();
  unsigned long long sum = 0, a, b;
  for (unsigned int i = 0; i < spins; i++) {
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(a) : "l"(w));
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(b) : "l"(w + 1));
    sum += a ^ b;
    if (i % 64 == 0) {
      asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(a) : "l"(w + 2)); sum += a;
      if (now() - began > budget) break;
    }
  }
  if (sum == 12345) o[0] = sum;
}
'''
prog = P()
ok(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"burn.cu", 0, None, None), "nvrtcCreateProgram")
opts = (ctypes.c_char_p * 1)(b"--gpu-architecture=compute_90")
ok(nvrtc.nvrtcCompileProgram(prog, 1, opts), "nvrtcCompileProgram")
size = S(); ok(nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size)), "ptx size")
ptx = ctypes.create_string_buffer(size.value); ok(nvrtc.nvrtcGetPTX(prog, ptx), "ptx")
ok(cuda.cuInit(0), "cuInit")
dev = ctypes.c_int(); ok(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = P(); ok(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "ctx"); ok(cuda.cuCtxSetCurrent(ctx), "set ctx")
mod = P(); ok(cuda.cuModuleLoadData(ctypes.byref(mod), ptx), "module")
fn = P(); ok(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, kind.encode()), "function")
bytes_ = 2 << 30
a, b = Q(), Q()
ok(cuda.cuMemAlloc_v2(ctypes.byref(a), bytes_), "alloc a"); ok(cuda.cuMemAlloc_v2(ctypes.byref(b), bytes_), "alloc b")
n, iters, spins, budget = Q(bytes_ // 16), ctypes.c_int(1 << 16), U(1 << 16), Q(100_000_000)
host = P(); ok(cuda.cuMemHostAlloc(ctypes.byref(host), 4096, 0x02), "host words"); ctypes.memset(host, 0, 4096)
words = Q(); ok(cuda.cuMemHostGetDevicePointer_v2(ctypes.byref(words), host, 0), "host words' device address")
at = ctypes.addressof
args = {
    "copy": lambda: (P * 4)(at(a), at(b), at(n), at(budget)),
    "spin": lambda: (P * 3)(at(a), at(iters), at(budget)),
    "pulse": lambda: (P * 4)(at(words), at(spins), at(b), at(budget)),
}[kind]()
grid, block = (1, 1) if kind == "pulse" else (48 * 8, 256)
deadline = time.time() + seconds
began, launches = time.time(), 0
while time.time() < deadline and not os.path.exists(stop):
    ok(cuda.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None, args, None), "launch")
    ok(cuda.cuCtxSynchronize(), "sync"); launches += 1
    if launches % 20 == 0:
        elapsed = time.time() - began
        print(f"burn {kind}: {launches / elapsed:.1f} launches/s" + {"copy": f", {2 * bytes_ * launches / elapsed / 1e9:.0f} GB/s moved", "pulse": f", {spins.value * launches / elapsed / 1e6:.1f} M spins/s"}.get(kind, ""), flush=True)
        began, launches = time.time(), 0
# Ended at a kernel boundary: nothing in flight reads what is freed.
ok(cuda.cuCtxSynchronize(), "final sync")
cuda.cuMemFreeHost(host)
print(f"burn {kind}: stopped", flush=True)
EOF
[ -z "${MRIG_DRY:-}" ] || exit 0
run() {
  "$BIN" --ignored --nocapture --exact "experts::source::tests::${1:-a_prompts_reads_land_in_a_reserved_bank_at_the_heaps_rate}" > "$BURN.reads.log" 2>&1 || true
  grep -E "reads |panicked|test result" "$BURN.reads.log" || { echo "no read lines; its output ends:"; tail -5 "$BURN.reads.log"; }
}
start() {
  python3 -X faulthandler -u "$BURN" "$1" 120 "$BURN.$1.stop" > "$BURN.$1.log" 2>&1 &
}
stop() {
  touch "$BURN.$1.stop"
}
for kind in copy spin pulse; do
  echo "== burner $kind alone"; python3 -X faulthandler -u "$BURN" "$kind" 8 "$BURN.$kind.stop" 2>&1 | tail -3
done
echo "== alone"; run
echo "== beside a served prompt's host work"; run a_prompts_reads_beside_a_served_prompts_host_work
for kinds in copy spin pulse "copy spin pulse"; do
  for kind in $kinds; do start "$kind"; done
  sleep 5
  for kind in $kinds; do grep -q "launches/s" "$BURN.$kind.log" || { echo "burn $kind is not running:"; cat "$BURN.$kind.log"; }; done
  echo "== beside a GPU $kinds"; run
  for kind in $kinds; do stop "$kind"; done
  wait || true
  for kind in $kinds; do tail -2 "$BURN.$kind.log"; done
  for kind in $kinds; do mv "$BURN.$kind.stop" "$BURN.$kind.stopped"; done
done
