import { Schema } from "effect"
import { describe, expect, it } from "vitest"
import { ModelInstance } from "@magnitudedev/icn-protocol/schemas"
import { projectInferenceResidency } from "./inference-projection"

const instance = (lifecycle: unknown) =>
  Schema.decodeUnknownSync(ModelInstance)({ id: "instance-1", modelId: "model:gguf:q4", lifecycle })

describe("inference residency projection", () => {
  it("keeps why a stopped instance stopped", () => {
    expect(projectInferenceResidency(instance({ _tag: "Stopped", reason: "memory_pressure" })))
      .toEqual({ _tag: "Stopped", reason: "memory_pressure" })
    expect(projectInferenceResidency(instance({ _tag: "Stopped", reason: "user_stop" })))
      .toEqual({ _tag: "Stopped", reason: "user_stop" })
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
    expect(failed({ _tag: "UnderPressure" })).toMatchObject({
      _tag: "Failed", failure: { shortage: { _tag: "UnderPressure" } },
    })
  })
})
