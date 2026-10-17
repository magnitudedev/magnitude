import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import { ActiveModelDeviceSchema } from "./acn-health"

describe("ActiveModelDeviceSchema", () => {
  it("accepts worker-confirmed CUDA placement and an optional human-readable device name", () => {
    expect(Schema.decodeUnknownSync(ActiveModelDeviceSchema)({
      instanceId: "instance-1",
      modelId: "minicpm5:gguf:q4",
      deviceId: "cuda:1",
      deviceName: "NVIDIA GeForce RTX 3050 Laptop GPU",
      backend: "cuda",
    })).toEqual({
      instanceId: "instance-1",
      modelId: "minicpm5:gguf:q4",
      deviceId: "cuda:1",
      deviceName: "NVIDIA GeForce RTX 3050 Laptop GPU",
      backend: "cuda",
    })
  })

  it("rejects unknown execution backends", () => {
    expect(() => Schema.decodeUnknownSync(ActiveModelDeviceSchema)({
      instanceId: "instance-1",
      modelId: "minicpm5:gguf:q4",
      deviceId: "cuda:1",
      backend: "unknown",
    })).toThrow()
  })
})
