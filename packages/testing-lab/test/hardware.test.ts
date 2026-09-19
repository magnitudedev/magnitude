import { expect, test } from "vitest"
import { Effect } from "effect"
import { targets } from "../src/catalog"
import { attestGeneration, attestHost, type BackendObservation, type HostObservation } from "../src/hardware"

const a10 = targets.find(t => t.os === "ubuntu" && t.backend === "cuda" && t.hardware === "a10")!
const host: HostObservation = { os: "ubuntu", version: "24.04.5", build: "fixture", arch: "x64", cpuVendor: "AuthenticAMD", cpuName: "AMD EPYC",
  machineModel: "Virtual Machine", memoryBytes: 128 * 1024 ** 3, gpus: [{ name: "NVIDIA A10", backend: "cuda", uuid: "fixture-gpu", driver: "fixture", memoryBytes: 24 * 1024 ** 3 }] }
const generation: BackendObservation = { backend: "cuda", model: "fixture-model", requestId: "fixture-request", loadedModuleDigest: "a".repeat(64),
  devices: [{ name: "NVIDIA A10", uuid: "fixture-gpu", allocatedModelBytes: 1024 }], offloadedLayers: 1, totalLayers: 1 }
test("host checks distinguish A10 from A100 and Intel from AMD", async () => {
  await Effect.runPromise(attestHost(a10, host))
  expect(await Effect.runPromise(attestHost(a10, { ...host, gpus: [{ ...host.gpus[0]!, name: "NVIDIA A100" }] }).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
  const intel = targets.find(t => t.os === "ubuntu" && t.hardware === "intel")!
  expect(await Effect.runPromise(attestHost(intel, host).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
})
test("hardware support alone cannot certify CUDA generation or mask CPU fallback", async () => {
  await Effect.runPromise(attestGeneration(a10, "fixture-model", "fixture-request", generation))
  for (const altered of [{ ...generation, backend: "cpu" as const }, { ...generation, devices: [] }, { ...generation, offloadedLayers: 0 }, { ...generation, requestId: "unrelated" }]) {
    expect(await Effect.runPromise(attestGeneration(a10, "fixture-model", "fixture-request", altered).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
  }
})
test("RTX PRO 6000 requires the actual Blackwell Server Edition device", async () => {
  const target = targets.find(t => t.os === "ubuntu" && t.backend === "cuda" && t.hardware === "rtx-pro-6000")!
  expect(await Effect.runPromise(attestHost(target, { ...host, gpus: [{ ...host.gpus[0]!, name: "Quadro RTX 6000" }] }).pipe(Effect.either))).toMatchObject({ _tag: "Left" })
  await Effect.runPromise(attestHost(target, { ...host, gpus: [{ ...host.gpus[0]!, name: "NVIDIA RTX PRO 6000 Blackwell Server Edition" }] }))
})
