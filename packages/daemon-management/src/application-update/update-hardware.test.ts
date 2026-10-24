import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { OwnedServiceState } from "@magnitudedev/sdk/desktop-host"
import { AcnReady, type AcnHardwareSummary } from "@magnitudedev/acn-protocol"
import { updateHardwareFromService, updateHardwareFromSummary } from "./update-hardware"

const ready = (hardware: Option.Option<AcnHardwareSummary>) => ({ _tag: "Ready", health: {
  service: "magnitude-acn", version: "0.2.9", revision: 1, id: "child", pid: 43, rpcVersion: 1, state: new AcnReady({ hardware }),
} }) as unknown as OwnedServiceState

describe("update hardware", () => {
  it("rounds memory to whole gigabytes and keeps labels printable and bounded", () => {
    expect(updateHardwareFromSummary({
      backend: "metal", gpuName: Option.some("Apple M1 Ultra"), gpuKind: Option.some("integrated"), gpuMemoryBytes: Option.some(137438953472),
      gpuCount: 1, gpuUnusedReason: Option.none(), cpuModel: Option.some("  Apple\tM1 Ultra é "), cpuCores: Option.some(20), memoryBytes: 137438953472,
    })).toEqual({
      backend: Option.some("metal"), gpu_name: Option.some("Apple M1 Ultra"), gpu_kind: Option.some("integrated"), gpu_memory_gb: Option.some(128), gpu_count: Option.some(1),
      gpu_unused_reason: Option.none(), cpu_model: Option.some("Apple M1 Ultra"), cpu_cores: Option.some(20), memory_gb: Option.some(128),
    })
    const long = updateHardwareFromSummary({ backend: "cpu", gpuName: Option.none(), gpuKind: Option.none(), gpuMemoryBytes: Option.none(), gpuCount: 1,
      gpuUnusedReason: Option.some("x".repeat(500)), cpuModel: Option.some("y".repeat(500)), cpuCores: Option.none(), memoryBytes: 8_000_000_000 })
    expect(Option.getOrThrow(long.gpu_unused_reason)).toHaveLength(200)
    expect(Option.getOrThrow(long.cpu_model)).toHaveLength(96)
    expect(long.memory_gb).toEqual(Option.some(7))
  })
  it("reports nothing until the service is ready with a hardware summary", () => {
    expect(updateHardwareFromService({ _tag: "Starting", attempt: 1, health: Option.none() } as unknown as OwnedServiceState)).toEqual(Option.none())
    expect(updateHardwareFromService(ready(Option.none()))).toEqual(Option.none())
    const reported = updateHardwareFromService(ready(Option.some({
      backend: "cuda", gpuName: Option.some("NVIDIA A10-8Q"), gpuKind: Option.some("discrete"), gpuMemoryBytes: Option.some(8589934592), gpuCount: 1,
      gpuUnusedReason: Option.none(), cpuModel: Option.some("AMD EPYC 7V13 64-Core Processor"), cpuCores: Option.some(12), memoryBytes: 118111600640 })))
    expect(Option.map(reported, hardware => [Option.getOrNull(hardware.backend), Option.getOrNull(hardware.gpu_memory_gb), Option.getOrNull(hardware.memory_gb)])).toEqual(Option.some(["cuda", 8, 110]))
  })
})
