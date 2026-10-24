import { Effect, Either, Option, Schema } from "effect"
import { expect, it } from "vitest"
import { HardwareSnapshot, ModelInstancesSnapshot } from "@magnitudedev/icn-protocol/schemas"
import { projectInferenceHardware, summarizeInferenceHardware } from "./inference-projection"

const store = (id: string) => ({
  id, total_capacity_bytes: 64, stable_capacity_bytes: 60, allocation_capacity_bytes: 64,
  planning_reserve_bytes: 4, emergency_reserve_bytes: 1, capacity_basis: "linux_mem_total",
})
const host = { store: store("system"), available_bytes: 40, current_free_bytes: 32 }
const executor = (id: string, backend: string, paths: number[], reason?: string) => ({
  id, backend, name: "Test GPU", description: "Test GPU", device_placement: paths, upload_placement: paths,
  availability: reason === undefined ? { type: "available", stable_capacity_bytes: 60 }
    : { type: "unavailable", reason },
})
const gpu = {
  id: "physical-gpu", kind: "integrated_gpu", executors: [executor("cuda:0", "cuda", [0]), executor("vulkan:0", "vulkan", [1])],
}
const cpu = { id: "host", kind: "cpu", executors: [executor("host-cpu", "cpu", [])] }
const paths = [
  { id: 0, executor: "cuda:0", store: "system", grant: null },
  { id: 1, executor: "vulkan:0", store: "system", grant: null },
]
const fields = (overrides: Record<string, unknown> = {}) => ({
  captured_at: 1, platform: "linux", architecture: "x86_64", cpu_model: "AMD EPYC 7742", logical_cores: 4,
  native_build: "test", topology_fingerprint: "test", host_ram: host, device_memory: [],
  volumes: [], physical_devices: [cpu], selected_executor: "host-cpu", access_paths: [], transfer_paths: [], ...overrides,
})
const snapshot = (overrides: Record<string, unknown> = {}) => Schema.decodeUnknownSync(HardwareSnapshot)(fields(overrides))
const instances = (...ready: { id: string; backend: string }[]) => Schema.decodeUnknownSync(ModelInstancesSnapshot)({
  revision: 1,
  instances: ready.map((device, index) => ({
    id: `instance-${index}`, modelId: "model",
    lifecycle: { _tag: "Ready", allocation: { contextWindowTokens: 4096, memoryDomains: [] }, device },
  })),
})
const none = instances()

it("preserves machine-wide physical core counts independently of process parallelism", () => {
  const value = Effect.runSync(projectInferenceHardware(snapshot({ physical_cores: 128 }), none))
  expect(value.physicalCores).toEqual(Option.some(128))
  expect(value.logicalCores).toBe(4)
  expect(value.processor).toEqual(Option.some("AMD EPYC 7742"))
})
it.each([undefined, null, 0])("keeps an unavailable physical core observation absent: %s", count => {
  expect(Effect.runSync(projectInferenceHardware(snapshot(count === undefined ? {} : { physical_cores: count }), none)).physicalCores).toEqual(Option.none())
})
it("groups backend executors by reported physical identity", () => {
  const value = Effect.runSync(projectInferenceHardware(snapshot({ physical_devices: [cpu, gpu], access_paths: paths }), none))
  expect(value.accelerators).toHaveLength(1)
  expect(value.accelerators[0]?.executors.map(e => e.deviceId)).toEqual(["cuda:0", "vulkan:0"])
  expect(value.memoryDomains).toHaveLength(1)
})
it("classifies host RAM as unified only when it is the active device's own memory", () => {
  const kind = (selected: string, current = none) => Effect.runSync(projectInferenceHardware(
    snapshot({ physical_devices: [cpu, gpu], selected_executor: selected, access_paths: paths }), current)).memoryDomains[0]?.kind
  // An integrated GPU that can place in host RAM does not make it unified while the CPU serves.
  expect(kind("host-cpu")).toBe("System")
  expect(kind("vulkan:0")).toBe("UnifiedMemory")
  // A resident model on the integrated GPU makes it the active device.
  expect(kind("host-cpu", instances({ id: "vulkan:0", backend: "vulkan" }))).toBe("UnifiedMemory")
})
it("retains both stores in a placement", () => {
  const deviceMemory = { physical_device_id: gpu.id, store: store("vram"), current_free_bytes: null }
  const value = Effect.runSync(projectInferenceHardware(snapshot({
    device_memory: [deviceMemory], physical_devices: [cpu, { ...gpu, kind: "gpu", executors: [executor("vulkan:0", "vulkan", [0, 1])] }],
    selected_executor: "vulkan:0", access_paths: [{ ...paths[1], id: 0, store: "vram" }, paths[1]],
  }), none))
  expect(value.accelerators[0]?.executors[0]?.memoryDomainIds).toEqual(["vram", "system"])
  expect(value.memoryDomains.map(domain => [domain.memoryDomainId, domain.kind, domain.availableBytes])).toEqual([
    ["system", "System", Option.some(32)],
    ["vram", "PhysicalDevice", Option.none()],
  ])
})
it("omits unavailable executors and CPU physical devices from accelerators", () => {
  const value = Effect.runSync(projectInferenceHardware(snapshot({
    physical_devices: [{ ...gpu, executors: [executor("cuda:0", "cuda", [], "driver too old"), gpu.executors[1]] },
      { id: "host", kind: "cpu", executors: [executor("host-cpu", "cpu", [])] }], access_paths: paths,
  }), none))
  expect(value.accelerators.map(a => a.executors.map(e => e.deviceId))).toEqual([["vulkan:0"]])
})
it("rejects missing placement references", () => {
  expect(Effect.runSync(Effect.either(projectInferenceHardware(snapshot({ physical_devices: [gpu] }), none)))._tag).toBe("Left")
  expect(Effect.runSync(Effect.either(summarizeInferenceHardware(snapshot({ physical_devices: [gpu] }), none)))._tag)
    .toBe("Left")
})
it("rejects an active device that is not reported", () => {
  const value = snapshot({ selected_executor: "cuda:9" })
  expect(Effect.runSync(Effect.either(projectInferenceHardware(value, none)))._tag).toBe("Left")
  expect(Effect.runSync(Effect.either(summarizeInferenceHardware(value, none)))._tag).toBe("Left")
  expect(Effect.runSync(Effect.either(summarizeInferenceHardware(snapshot(), instances({ id: "cuda:9", backend: "cuda" }))))._tag)
    .toBe("Left")
})
it("rejects a device placement on a store that is not a memory domain", () => {
  const volume = { store: store("volume"), current_free_bytes: 8 }
  const value = projectInferenceHardware(snapshot({
    volumes: [volume], physical_devices: [gpu],
    access_paths: [paths[0], { ...paths[1], store: "volume" }],
  }), none)
  expect(Effect.runSync(Effect.either(value))._tag).toBe("Left")
})
it("admits no snapshot without host RAM and its free reading, or without a serving device", () => {
  const { host_ram: _, ...withoutHost } = fields()
  const { selected_executor: __, ...withoutServing } = fields()
  for (const invalid of [withoutHost, fields({ host_ram: { store: host.store } }), withoutServing, fields({ selected_executor: null })]) {
    expect(Either.isLeft(Schema.decodeUnknownEither(HardwareSnapshot)(invalid))).toBe(true)
  }
})
it("summarizes the serving device and counts one GPU across backend views", () => {
  const summary = Effect.runSync(summarizeInferenceHardware(
    snapshot({ physical_devices: [gpu], selected_executor: "cuda:0", access_paths: paths, physical_cores: 8 }), none))
  expect(summary).toMatchObject({ backend: "cuda", gpuName: Option.some("Test GPU"), gpuKind: Option.some("integrated"),
    gpuCount: 1, gpuMemoryBytes: Option.some(64), cpuCores: Option.some(8), memoryBytes: 64 })
})
it("names a resident model's device, and otherwise the serving device", () => {
  const discrete = { id: "physical-dgpu", kind: "gpu", executors: [{ ...executor("cuda:1", "cuda", [2]), name: "Discrete GPU" }] }
  const hybrid = snapshot({
    physical_devices: [cpu, gpu, discrete], selected_executor: "vulkan:0",
    access_paths: [...paths, { id: 2, executor: "cuda:1", store: "vram", grant: null }],
    device_memory: [{ physical_device_id: discrete.id, store: store("vram"), current_free_bytes: null }],
  })
  expect(Effect.runSync(summarizeInferenceHardware(hybrid, none))).toMatchObject({ backend: "vulkan", gpuName: Option.some("Test GPU") })
  const resident = instances({ id: "cuda:1", backend: "cuda" })
  expect(Effect.runSync(summarizeInferenceHardware(hybrid, resident)))
    .toMatchObject({ backend: "cuda", gpuName: Option.some("Discrete GPU"), gpuKind: Option.some("discrete") })
  expect(Effect.runSync(projectInferenceHardware(hybrid, resident)).activeDeviceId).toBe("cuda:1")
  expect(Effect.runSync(projectInferenceHardware(hybrid, none)).activeDeviceId).toBe("vulkan:0")
})
it("reports a GPU's memory as its own memory's total, or what its grant lets it place in host RAM", () => {
  const vram = { physical_device_id: gpu.id, store: { ...store("vram"), total_capacity_bytes: 24 }, current_free_bytes: null }
  const gpuMemory = (accessPaths: unknown[], deviceMemory: unknown[] = []) => Effect.runSync(summarizeInferenceHardware(
    snapshot({ physical_devices: [{ ...gpu, executors: [executor("vulkan:0", "vulkan", [0, 1])] }], selected_executor: "vulkan:0",
      access_paths: accessPaths, device_memory: deviceMemory }), none)).gpuMemoryBytes
  const granted = (id: number, bytes: number) => ({ id, executor: "vulkan:0", store: "system",
    grant: { bytes, basis: "vulkan_heaps", remaining_bytes: null } })
  // Two host heaps share the driver's total.
  expect(gpuMemory([granted(0, 10), granted(1, 20)])).toEqual(Option.some(30))
  // Its own memory, then host RAM beyond it.
  expect(gpuMemory([{ id: 0, executor: "vulkan:0", store: "vram", grant: null }, granted(1, 8)], [vram]))
    .toEqual(Option.some(32))
})
it("explains an unused GPU from its own reason, then the backend diagnostic, and otherwise as not selected", () => {
  const reason = (executors: unknown[], overrides: Record<string, unknown> = {}) => Effect.runSync(summarizeInferenceHardware(
    snapshot({ physical_devices: [cpu, { ...gpu, executors }], access_paths: paths, ...overrides }), none)).gpuUnusedReason
  expect(reason([executor("cuda:0", "cuda", [], "driver too old")])).toEqual(Option.some("driver too old"))
  const diagnostic = { source: { type: "backend", backend: "cuda" }, message: "no driver" }
  expect(reason([gpu.executors[0]], { backend_diagnostics: [diagnostic] })).toEqual(Option.some("cuda: no driver"))
  expect(reason([gpu.executors[0]])).toEqual(Option.some("cuda: not selected"))
})
it("reports no GPU fields on a CPU-only machine", () => {
  const summary = Effect.runSync(summarizeInferenceHardware(snapshot({ cpu_model: null }), none))
  expect(summary).toEqual({ backend: "cpu", gpuName: Option.none(), gpuKind: Option.none(), gpuMemoryBytes: Option.none(),
    gpuCount: 0, gpuUnusedReason: Option.none(), cpuModel: Option.none(), cpuCores: Option.some(4), memoryBytes: 64 })
})
