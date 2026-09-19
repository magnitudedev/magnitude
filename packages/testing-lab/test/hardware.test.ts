import { expect, test } from "vitest"
import { Effect, Schema } from "effect"
import { NativeExecution } from "../src/execution-telemetry"
import { targets } from "../src/catalog"
import { attestGeneration, attestHost, type HostObservation } from "../src/hardware"

const a10 = targets.find(t => t.os === "ubuntu" && t.backend === "cuda" && t.hardware === "a10")!
const host: HostObservation = { os: "ubuntu", version: "24.04.5", build: "fixture", arch: "x64", cpuVendor: "AuthenticAMD", cpuName: "AMD EPYC",
  machineModel: "Virtual Machine", memoryBytes: 128 * 1024 ** 3, gpus: [{ name: "NVIDIA A10", backend: "cuda", uuid: "fixture-gpu", driver: "fixture", memoryBytes: 24 * 1024 ** 3 }] }
const allocation = { kind: "device" as const, backend: "CUDA", physical_id: "fixture-gpu", native_index: 0, model_bytes: 1024 }
const generation = Schema.decodeUnknownSync(NativeExecution)({ traceId: "a".repeat(32), model: "fixture-model", requestId: "7", workerPid: 42, workerGeneration: "1", allocations: [allocation] })
test("host checks distinguish A10 from A100 and Intel from AMD", async () => {
  await Effect.runPromise(attestHost(a10, host))
  expect(await Effect.runPromise(attestHost(a10, { ...host, gpus: [{ ...host.gpus[0]!, name: "NVIDIA A100" }] }).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
  const intel = targets.find(t => t.os === "ubuntu" && t.hardware === "intel")!
  expect(await Effect.runPromise(attestHost(intel, host).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
})
test("hardware support alone cannot certify CUDA generation or mask CPU fallback", async () => {
  await Effect.runPromise(attestGeneration(a10, host, "fixture-model", generation))
  for (const altered of [{ ...generation, allocations: [{ kind: "host" as const, model_bytes: 1024 }] }, { ...generation, allocations: [] }, { ...generation, allocations: [{ ...allocation, physical_id: null }] }, { ...generation, model: "unrelated" }, { ...generation, allocations: [{ ...allocation, backend: "MTL" }] }, { ...generation, allocations: [{ ...allocation, physical_id: "other-gpu" }] }]) {
    expect(await Effect.runPromise(attestGeneration(a10, host, "fixture-model", altered).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
  }
})
test("CPU device allocations qualify CPU execution without hiding accelerator fallback or mixed backends", async () => {
  const cpu = targets.find(t => t.os === "ubuntu" && t.backend === "cpu" && t.hardware === "amd")!
  const cpuDevice = { ...allocation, backend: "CPU", physical_id: null, model_bytes: 2959104000 }
  const observed = { ...generation, allocations: [cpuDevice, { kind: "host" as const, model_bytes: 2904582144 }] }
  await Effect.runPromise(attestGeneration(cpu, host, "fixture-model", observed))
  await Effect.runPromise(attestGeneration(cpu, host, "fixture-model", { ...generation, allocations: [cpuDevice] }))
  await Effect.runPromise(attestGeneration(a10, host, "fixture-model", { ...observed, allocations: [...observed.allocations, allocation] }))
  for (const invalid of [
    attestGeneration(a10, host, "fixture-model", observed),
    attestGeneration(cpu, host, "fixture-model", { ...observed, allocations: [...observed.allocations, allocation] }),
    attestGeneration(cpu, host, "fixture-model", { ...generation, allocations: [{ ...cpuDevice, backend: "unknown" }] }),
  ]) expect((await Effect.runPromise(invalid.pipe(Effect.either)))._tag).toBe("Left")
})
test("RTX PRO 6000 requires the actual Blackwell Server Edition device", async () => {
  const target = targets.find(t => t.os === "ubuntu" && t.backend === "cuda" && t.hardware === "rtx-pro-6000")!
  expect(await Effect.runPromise(attestHost(target, { ...host, gpus: [{ ...host.gpus[0]!, name: "Quadro RTX 6000" }] }).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
  await Effect.runPromise(attestHost(target, { ...host, gpus: [{ ...host.gpus[0]!, name: "NVIDIA RTX PRO 6000 Blackwell Server Edition" }] }))
})

test("Metal without a physical ID requires exactly one device at native index zero", () => Effect.runPromise(Effect.gen(function* () {
  const target = targets.find(t => t.os === "macos" && t.version === "15" && t.backend === "metal")!
  const apple: HostObservation = { ...host, os: "macos", version: "15.5", arch: "arm64", cpuVendor: "Apple", cpuName: "Apple M4",
    gpus: [{ ...host.gpus[0]!, backend: "metal", name: "Apple M4", uuid: "apple-device" }] }
  const observed = { ...generation, allocations: [{ ...allocation, backend: "MTL", physical_id: null }] }
  yield* attestGeneration(target, apple, "fixture-model", observed)
  for (const invalid of [
    attestGeneration(target, { ...apple, gpus: [...apple.gpus, { ...apple.gpus[0]!, uuid: "second" }] }, "fixture-model", observed),
    attestGeneration(target, apple, "fixture-model", { ...observed, allocations: [{ ...observed.allocations[0]!, native_index: 1 }] }),
  ]) expect((yield* invalid.pipe(Effect.either))._tag).toBe("Left")
  const cpu = targets.find(t => t.os === "macos" && t.version === "15" && t.backend === "cpu")!
  expect((yield* attestGeneration(cpu, apple, "fixture-model", observed).pipe(Effect.either))._tag).toBe("Left")
  yield* attestGeneration(cpu, apple, "fixture-model", { ...generation, allocations: [{ kind: "host", model_bytes: 1024 }] })
})))
