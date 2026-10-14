import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { HardwareSnapshot } from "@magnitudedev/icn-protocol/schemas"
import type * as InferenceSchema from "@magnitudedev/icn-protocol/schemas"
import { projectInferenceHardware, summarizeInferenceHardware } from "./inference-projection"

const snapshot = (physical?: number | null, memoryDomains: readonly unknown[] = []) => Schema.decodeUnknownSync(HardwareSnapshot)({
  captured_at: 1, platform: "linux", architecture: "x86_64", cpu_model: "AMD EPYC 7742",
  logical_cores: 4, ...(physical === undefined ? {} : { physical_cores: physical }),
  system_memory: { physical_capacity_bytes: 64, physical_available_bytes: 32, allocation_capacity_bytes: 64,
    allocation_headroom_bytes: 32, assess_reserve_bytes: 1, abort_reserve_bytes: 1 },
  native_build: "test", enabled_backends: [], topology_fingerprint: "test", memory_domains: memoryDomains,
})
it("preserves machine-wide physical topology independently of process parallelism", () => {
  const value = Effect.runSync(projectInferenceHardware(snapshot(128)))
  expect(value.physicalCores).toEqual(Option.some(128))
  expect(value.logicalCores).toBe(4)
  expect(value.processor).toEqual(Option.some("AMD EPYC 7742"))
})
it.each([undefined, null, 0])("keeps an unavailable physical core observation absent: %s", count => {
  expect(Effect.runSync(projectInferenceHardware(snapshot(count))).physicalCores).toEqual(Option.none())
})
it("lists only devices a model can run on as accelerators", () => {
  const device = (id: string, kind: string, reason?: string) => ({
    id, backend: id === "host-cpu" ? "cpu" : "vulkan", name: id, description: id, kind,
    ...(reason === undefined ? {} : { unavailable_reason: reason }),
  })
  const value = Effect.runSync(projectInferenceHardware(snapshot(8, [{
    id: "system", kind: "system", total_capacity_bytes: 64, stable_capacity_bytes: 32, shares_system_memory: true,
    devices: [
      device("host-cpu", "cpu"),
      device("vulkan:00000000-0000-0000-0000-000000000001", "integrated_gpu",
        "it lacks VK_KHR_shader_fma and its fma is not fused beside a NoContraction multiply-add"),
      device("vulkan:00000000-0000-0000-0000-000000000002", "integrated_gpu"),
    ],
  }])))
  expect(value.accelerators.map(({ acceleratorId }) => acceleratorId))
    .toEqual(["vulkan:00000000-0000-0000-0000-000000000002"])
})

describe("summarizeInferenceHardware", () => {
  const device = (fields: Partial<InferenceSchema.HardwareDevice> & Pick<InferenceSchema.HardwareDevice, "id" | "backend" | "name" | "kind">): InferenceSchema.HardwareDevice => ({
    description: fields.name, memory_limit: Option.none(), unavailable_reason: Option.none(), ...fields,
  })
  const domain = (fields: Partial<InferenceSchema.HardwareMemoryDomain> & Pick<InferenceSchema.HardwareMemoryDomain, "id" | "kind" | "devices">): InferenceSchema.HardwareMemoryDomain => ({
    current_free_bytes: Option.none(), shares_system_memory: fields.kind !== "physical_device", stable_capacity_bytes: 0, total_capacity_bytes: 0, ...fields,
  })
  const snapshot = (fields: Partial<InferenceSchema.HardwareSnapshot>): InferenceSchema.HardwareSnapshot => ({
    architecture: "x86_64", captured_at: 1, cpu_model: Option.some("AMD EPYC 7V13 64-Core Processor"), enabled_backends: ["cpu", "cuda", "vulkan"],
    logical_cores: 12, memory_domains: [], native_build: "test", physical_cores: Option.some(6), platform: "windows",
    system_memory: { abort_reserve_bytes: 0, allocation_capacity_bytes: 0, allocation_headroom_bytes: 0, assess_reserve_bytes: 0, physical_available_bytes: 0, physical_capacity_bytes: 118111600640 },
    system_product_name: Option.none(), topology_fingerprint: "f", backend_diagnostics: Option.none(), ...fields,
  })
  const cudaCard = domain({ id: "gpu-1", kind: "physical_device", total_capacity_bytes: 8589934592, devices: [
    device({ id: "cuda:0", backend: "cuda", name: "NVIDIA A10-8Q", kind: "gpu", memory_limit: Option.some({ kind: "recommended_working_set", total_bytes: 8589934592, stable_bytes: 8000000000, current_free_bytes: Option.none() }) }),
    device({ id: "vulkan:0", backend: "vulkan", name: "NVIDIA A10-8Q", kind: "gpu" }),
  ] })

  it("describes the GPU the loaded backend runs on and counts a card once across backends", () => {
    const summary = summarizeInferenceHardware(Option.some({ _tag: "Cuda", hardwareLabel: "NVIDIA A10-8Q" }), snapshot({ memory_domains: [cudaCard] }))
    expect(summary).toEqual({ backend: "cuda", gpuName: Option.some("NVIDIA A10-8Q"), gpuKind: Option.some("discrete"), gpuMemoryBytes: Option.some(8589934592), gpuCount: 1,
      gpuUnusedReason: Option.none(), cpuModel: Option.some("AMD EPYC 7V13 64-Core Processor"), cpuCores: Option.some(6), memoryBytes: 118111600640 })
  })
  it("explains an unused GPU from its own reason, then the backend diagnostic, then the selection policy", () => {
    const unavailable = domain({ id: "gpu-1", kind: "physical_device", total_capacity_bytes: 8589934592, devices: [
      device({ id: "cuda:0", backend: "cuda", name: "NVIDIA A10-8Q", kind: "gpu", unavailable_reason: Option.some("its CUDA context could not be created: driver too old") }) ] })
    expect(summarizeInferenceHardware(Option.some({ _tag: "Cpu", hardwareLabel: "CPU" }), snapshot({ memory_domains: [unavailable] })).gpuUnusedReason)
      .toEqual(Option.some("its CUDA context could not be created: driver too old"))
    const onVulkan = summarizeInferenceHardware(Option.some({ _tag: "Vulkan", hardwareLabel: "NVIDIA A10-8Q" }), snapshot({ memory_domains: [cudaCard],
      backend_diagnostics: Option.some([{ backend: "cuda", message: "no CUDA-capable device is detected" }]) }))
    expect([onVulkan.backend, Option.getOrNull(onVulkan.gpuName), Option.getOrNull(onVulkan.gpuUnusedReason)]).toEqual(["vulkan", "NVIDIA A10-8Q", null])
    const integrated = domain({ id: "system", kind: "system", total_capacity_bytes: 34359738368, devices: [
      device({ id: "vulkan:0", backend: "vulkan", name: "Intel(R) Iris(R) Xe Graphics", kind: "integrated_gpu" }) ] })
    const onCpu = summarizeInferenceHardware(Option.some({ _tag: "Cpu", hardwareLabel: "CPU" }), snapshot({ memory_domains: [integrated] }))
    expect([onCpu.backend, Option.getOrNull(onCpu.gpuName), onCpu.gpuCount, Option.getOrNull(onCpu.gpuUnusedReason)])
      .toEqual(["cpu", null, 1, "vulkan: integrated GPU is not selected automatically"])
    const diagnosed = summarizeInferenceHardware(Option.some({ _tag: "Cpu", hardwareLabel: "CPU" }), snapshot({ memory_domains: [cudaCard],
      backend_diagnostics: Option.some([{ backend: "cuda", message: "no CUDA-capable device is detected" }, { backend: "vulkan", message: "no Vulkan loader" }]) }))
    expect(Option.getOrNull(diagnosed.gpuUnusedReason)).toBe("cuda: no CUDA-capable device is detected")
  })
  it("falls back to logical cores and reports no GPU fields on a machine without one", () => {
    const summary = summarizeInferenceHardware(Option.none(), snapshot({ physical_cores: Option.none(), cpu_model: Option.none() }))
    expect(summary).toEqual({ backend: "cpu", gpuName: Option.none(), gpuKind: Option.none(), gpuMemoryBytes: Option.none(), gpuCount: 0,
      gpuUnusedReason: Option.none(), cpuModel: Option.none(), cpuCores: Option.some(12), memoryBytes: 118111600640 })
  })
})
