import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import { Generated } from "@magnitudedev/icn"
import { hostToWire } from "./service"

describe("local inference hardware projection", () => {
  it("projects Apple unified memory once and uses accelerator descriptions", () => {
    const gib = 1024 ** 3
    const hardware = Schema.decodeUnknownSync(Generated.HardwareSnapshotSchema)({
      captured_at: 1,
      platform: "macos",
      architecture: "aarch64",
      cpu_model: "Apple M4 Max",
      logical_cores: 16,
      system_memory: {
        total_bytes: 64 * gib,
        current_available_bytes: 40 * gib,
      },
      native_build: "test",
      enabled_backends: ["CPU", "MTL"],
      assessment_policy: "test",
      capacity_policy: "test",
      topology_fingerprint: "test",
      memory_domains: [{
        id: "system",
        kind: "unified_memory",
        total_capacity_bytes: 64 * gib,
        stable_capacity_bytes: 62.5 * gib,
        current_free_bytes: 40 * gib,
        shares_system_memory: true,
        devices: [{
          id: "cpu",
          backend: "CPU",
          name: "CPU",
          description: "Apple M4 Max",
          kind: "cpu",
          memory_limit: null,
        }, {
          id: "metal",
          backend: "MTL",
          name: "MTL0",
          description: "Apple M4 Max",
          kind: "gpu",
          memory_limit: {
            kind: "recommended_working_set",
            total_bytes: 48 * gib,
            stable_bytes: 46.5 * gib,
            current_free_bytes: 30 * gib,
          },
        }],
      }],
    })

    expect(hostToWire(hardware)).toEqual({
      platform: "macos",
      architecture: "aarch64",
      systemMemoryBytes: 64 * gib,
      cpuModel: "Apple M4 Max",
      logicalCores: 16,
      memoryDomains: [{
        id: "system",
        kind: "unified_memory",
        totalCapacityBytes: 64 * gib,
        stableCapacityBytes: 62.5 * gib,
        currentFreeBytes: 40 * gib,
        sharesSystemMemory: true,
        backendNames: ["Metal"],
        deviceNames: ["Apple M4 Max"],
        splitGroupId: null,
      }],
    })
  })
})
