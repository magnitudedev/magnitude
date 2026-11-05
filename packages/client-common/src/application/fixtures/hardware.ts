import { Schema } from "effect"
import { LocalInferenceHardwareSchema, type LocalInferenceHardware } from "@magnitudedev/sdk"

const GiB = 1024 ** 3

/**
 * A machine a streamed load serves on: a discrete "Test GPU" with its own memory ("cuda-0") beside
 * system memory ("system"), or a GPU whose memory is unified with the host's ("system").
 */
export const servingHardware = (memory: "Discrete" | "Unified"): LocalInferenceHardware =>
  Schema.decodeUnknownSync(LocalInferenceHardwareSchema)({
    platform: memory === "Discrete" ? "Linux" : "MacOS",
    architecture: memory === "Discrete" ? "X64" : "Arm64",
    logicalCores: 8,
    totalSystemMemoryBytes: 16 * GiB,
    accelerators: [{
      acceleratorId: "gpu",
      name: "Test GPU",
      executors: [{
        deviceId: "primary",
        backend: memory === "Discrete" ? "cuda" : "metal",
        memoryDomainIds: [memory === "Discrete" ? "cuda-0" : "system"],
      }],
    }],
    activeDeviceId: "primary",
    memoryDomains: [
      { memoryDomainId: "system", kind: memory === "Discrete" ? "System" : "UnifiedMemory", totalBytes: 16 * GiB, availableBytes: 8 * GiB },
      ...(memory === "Discrete" ? [{ memoryDomainId: "cuda-0", kind: "PhysicalDevice", totalBytes: 24 * GiB, availableBytes: 8 * GiB }] : []),
    ],
  })
