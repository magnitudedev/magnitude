import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Option, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import { join, resolve } from "node:path"
import pins from "../tools/cuda-toolkit.json"
import { Digest, InfrastructureFailure } from "../src/domain"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"

const Component = Schema.Struct({ name: Schema.String.pipe(Schema.pattern(/^[a-z_]+$/)),
  url: Schema.String.pipe(Schema.startsWith("https://developer.download.nvidia.com/compute/cuda/redist/")),
  sha256: Digest, bytes: Schema.Int.pipe(Schema.between(1, 2 * 1024 ** 3)) })
const fail = (message: string) => new InfrastructureFailure({ operation: "cuda-toolkit", message })

/** A user-owned compiler SDK; this never installs a driver or requires a GPU. */
export const prepareCudaToolkit = (destination: string) => Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const host = process.platform === "win32" && process.arch === "x64" ? "windows-x64-msvc"
    : process.platform === "linux" && process.arch === "x64" ? "linux-x64-gnu"
    : process.platform === "linux" && process.arch === "arm64" ? "linux-arm64-gnu" : undefined
  if (!host) return yield* fail("CUDA compilation requires a qualified Linux or Windows host")
  const components = yield* Schema.decodeUnknown(Schema.NonEmptyArray(Component))(pins.hosts[host])
  const root = resolve(destination)
  if (yield* fs.exists(root)) return yield* fail("CUDA toolkit requires a fresh owned directory")
  yield* fs.makeDirectory(root, { recursive: true, mode: 0o700 })
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "lab-cuda-toolkit-" })
  yield* fs.makeDirectory(join(root, "licenses"))
  for (const component of components) {
    const archive = join(temporary, component.name + (host === "windows-x64-msvc" ? ".zip" : ".tar.xz"))
    yield* checkedCommand(process.platform === "win32" ? "curl.exe" : "curl", ["--fail", "--silent", "--show-error", "--proto", "=https",
      "--max-redirs", "0", "--max-filesize", String(component.bytes), "--output", archive, component.url], { timeoutMs: 15 * 60_000 })
    if (Number((yield* fs.stat(archive)).size) !== component.bytes) return yield* fail(`CUDA ${component.name} download length mismatch`)
    const hash = createHash("sha256")
    yield* fs.stream(archive).pipe(Stream.runForEach(chunk => Effect.sync(() => { hash.update(chunk) })))
    if (hash.digest("hex") !== component.sha256) return yield* fail(`CUDA ${component.name} download digest mismatch`)
    // Administrator-pinned NVIDIA redistributables all have one top-level archive directory.
    yield* checkedCommand(process.platform === "win32" ? "tar.exe" : "tar", ["-xf", archive, "--strip-components=1", "-C", root], { timeoutMs: 10 * 60_000 })
    for (const license of ["LICENSE", "LICENSE.txt"]) if (yield* fs.exists(join(root, license))) {
      yield* fs.rename(join(root, license), join(root, "licenses", `${component.name}-${license}`))
    }
    yield* fs.remove(archive)
  }
  // The release pack's loader resolves Unix libraries through the conventional CUDA lib64 path.
  if (process.platform === "linux" && !(yield* fs.exists(join(root, "lib64")))) yield* fs.symlink("lib", join(root, "lib64"))
  const suffix = process.platform === "win32" ? ".exe" : ""
  const compiler = join(root, "bin", `nvcc${suffix}`)
  const version = yield* checkedCommand(compiler, ["--version"])
  if (!version.stdout.includes("release 12.9") || !version.stdout.includes("V12.9.86")) return yield* fail("CUDA compiler differs from the pinned 12.9 release")
  yield* checkedCommand(join(root, "bin", `cuobjdump${suffix}`), ["--version"])
  for (const header of ["cuda_runtime.h", "cublas_v2.h"]) if (!(yield* fs.exists(join(root, "include", header)))) return yield* fail(`CUDA header ${header} is missing`)
  const source = join(temporary, "probe.cu"), ptx = join(temporary, "probe.ptx")
  yield* fs.writeFileString(source, '#include <cublas_v2.h>\n__global__ void magnitude_lab_kernel(float *value) { value[threadIdx.x] += 1.0f; }\nextern "C" int magnitude_lab_link() { cublasHandle_t handle; return int(cublasCreate(&handle)); }\n')
  yield* checkedCommand(compiler, ["--ptx", "--gpu-architecture=compute_80", source, "-o", ptx], { timeoutMs: 120_000, cwd: Option.some(root) })
  const assembly = yield* fs.readFileString(ptx)
  if (!assembly.includes(".entry magnitude_lab_kernel") && !assembly.includes("magnitude_lab_kernel")) return yield* fail("CUDA compiler did not emit the probe kernel")
  yield* checkedCommand(compiler, ["--shared", "--cudart", "shared", "--gpu-architecture=sm_80",
    ...(process.platform === "linux" ? ["-Xcompiler", "-fPIC"] : []), source,
    "-L", join(root, process.platform === "win32" ? "lib/x64" : "lib64"), "-lcublas", "-o", join(temporary, process.platform === "win32" ? "probe.dll" : "probe.so")],
    { timeoutMs: 120_000, cwd: Option.some(root) })
  const receipt = yield* Schema.encode(Schema.parseJson(Schema.Struct({ version: Schema.String, host: Schema.String,
    compiler: Schema.String, components: Schema.Array(Component) })) )({ version: pins.version, host, compiler: version.stdout.trim(), components })
  yield* fs.writeFileString(join(root, "receipt.json"), receipt, { mode: 0o600 })
  yield* Console.log(receipt)
}))
if (import.meta.main) BunRuntime.runMain(Effect.gen(function* () {
  if (process.argv.length !== 3) return yield* fail("Expected an owned CUDA toolkit destination")
  yield* prepareCudaToolkit(process.argv[2]!)
}).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
