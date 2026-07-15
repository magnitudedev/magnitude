import { describe, expect, test } from "vitest"
import { Schema } from "effect"
import {
  LocalInferenceCapabilities,
  LocalModelDownloadProgress,
} from "./local-inference"

describe("local inference protocol schemas", () => {
  test("accepts stable capacity plus transient diagnostic memory", () => {
    const decoded = Schema.decodeUnknownSync(LocalInferenceCapabilities)({
      binary: { identity: "managed-llama-server" },
      system: { totalMemoryBytes: 64 * 1024 ** 3, logicalCores: 12 },
      accelerators: [{
        id: "CUDA0",
        backend: "CUDA",
        description: "consumer RTX",
        capacityBytes: 24 * 1024 ** 3,
        capacityKind: "physical-device-memory",
        memoryDomainId: "pci:1",
        sharesSystemMemory: false,
        currentFreeBytes: 1,
      }],
      warnings: [],
    })
    expect(decoded.accelerators[0]?.currentFreeBytes).toBe(1)
  })

  test("rejects negative byte counts", () => {
    expect(() => Schema.decodeUnknownSync(LocalModelDownloadProgress)({
      operationId: "operation",
      status: "downloading",
      completedBytes: -1,
      totalBytes: 100,
      resumable: true,
    })).toThrow()
  })
})

