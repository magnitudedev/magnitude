import { Effect, Option, Schema } from "effect"
import { expect, it } from "vitest"
import { HardwareSnapshot } from "@magnitudedev/icn-protocol/schemas"
import { projectInferenceHardware } from "./inference-projection"

const snapshot = (physical?: number | null) => Schema.decodeUnknownSync(HardwareSnapshot)({
  captured_at: 1, platform: "linux", architecture: "x86_64", cpu_model: "AMD EPYC 7742",
  logical_cores: 4, ...(physical === undefined ? {} : { physical_cores: physical }),
  system_memory: { physical_capacity_bytes: 64, physical_available_bytes: 32, allocation_capacity_bytes: 64,
    allocation_headroom_bytes: 32, assess_reserve_bytes: 1, abort_reserve_bytes: 1 },
  native_build: "test", enabled_backends: [], topology_fingerprint: "test", memory_domains: [],
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
