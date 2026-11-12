import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { ModelInstance } from "@magnitudedev/icn-protocol/schemas"
import { projectInferenceResidency } from "./inference-projection"
import { ModelResidencySchema } from "./schemas/model-state"

const instance = (lifecycle: unknown) =>
  Schema.decodeUnknownSync(ModelInstance)({ id: "instance-1", modelId: "model:gguf:q4", lifecycle })

/** How a load serves, as the service contract carries it. */
const serving = { promptGroupPlaces: { held: 3, capacity: 8 } } as const

describe("inference residency projection", () => {
  it("keeps why a stopped instance stopped", () => {
    expect(projectInferenceResidency(instance({ _tag: "Stopped", reason: "memory_pressure" })))
      .toEqual({ _tag: "Stopped", reason: "memory_pressure" })
    expect(projectInferenceResidency(instance({ _tag: "Stopped", reason: "user_stop" })))
      .toEqual({ _tag: "Stopped", reason: "user_stop" })
  })

  it("carries the device a ready instance runs on, and how it serves", () => {
    expect(projectInferenceResidency(instance({
      _tag: "Ready", allocation: { contextWindowTokens: 4096, memoryDomains: [], ...serving }, device: { id: "cuda:0", backend: "cuda" },
    }))).toEqual({
      _tag: "Ready",
      allocation: { contextWindowTokens: 4096, memoryDomains: [], ...serving, streaming: Option.none() },
      device: { deviceId: "cuda:0", backend: "cuda" },
    })
  })

  it("carries a memory shortage with its form", () => {
    const failed = (shortage: unknown) => projectInferenceResidency(instance({
      _tag: "Failed",
      failure: { _tag: "MemoryShortage", code: "memory_shortage", message: "short", retryable: true, shortage },
    }))
    expect(failed({ _tag: "Blocked", requiredBytes: 30, availableBytes: 24 })).toEqual({
      _tag: "Failed",
      failure: {
        _tag: "MemoryShortage", code: "memory_shortage", message: "short", retryable: true,
        shortage: { _tag: "Blocked", requiredBytes: 30, availableBytes: 24 },
      },
    })
  })

  it("carries where a streamed load's experts are, in the service contract's wire shape", () => {
    const streaming = {
      experts: 256,
      slots: 64,
      floorSlots: 26,
      ceilingSlots: 256,
      tiers: [{ memoryDomainId: "cuda-0", bytes: 300 }, { memoryDomainId: "system", bytes: 400 }],
      host: {
        memoryDomainId: "system", slots: 32, ceilingSlots: 64, bytes: 210,
        lastResize: { _tag: "Shrank", to: 32, by: 1, atUnixMs: 1234 },
      },
      diskBytes: 300,
      tables: [{
        records: 40, floorRecords: 1, ceilingRecords: 42, bytes: 500,
        lastResize: { _tag: "Grew", from: 39, by: 1, atUnixMs: 1200 },
      }],
      lastResize: { _tag: "Shrank", to: 64, by: 2, atUnixMs: 1100 },
    }
    const ready = projectInferenceResidency(instance({
      _tag: "Ready",
      allocation: { contextWindowTokens: 4096, memoryDomains: [], ...serving, streaming },
      device: { id: "cuda:0", backend: "cuda" },
    }))
    const encoded = Schema.encodeSync(ModelResidencySchema)(ready)
    expect(encoded).toEqual({
      _tag: "Ready",
      allocation: { contextWindowTokens: 4096, memoryDomains: [], ...serving, streaming },
      device: { deviceId: "cuda:0", backend: "cuda" },
    })
    expect(Schema.decodeUnknownSync(ModelResidencySchema)(encoded)).toEqual(ready)
  })

  it("leaves a load without an expert bank without streaming status", () => {
    const ready = projectInferenceResidency(instance({
      _tag: "Ready",
      allocation: { contextWindowTokens: 4096, memoryDomains: [], ...serving },
      device: { id: "cuda:0", backend: "cuda" },
    }))
    expect(ready).toMatchObject({ allocation: { streaming: Option.none() } })
  })
})
