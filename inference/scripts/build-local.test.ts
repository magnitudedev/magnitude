import { describe, expect, it } from "vitest"
import { selectLocalIcnBackend } from "./build-local"

const environment = (
  overrides: Partial<Parameters<typeof selectLocalIcnBackend>[0]> = {},
): Parameters<typeof selectLocalIcnBackend>[0] => ({
  platform: "linux",
  arch: "arm64",
  requested: undefined,
  nvidiaDriverAvailable: false,
  cudaToolkitAvailable: false,
  ...overrides,
})

describe("local ICN backend selection", () => {
  it("selects CUDA automatically when the driver and toolkit are available", () => {
    expect(selectLocalIcnBackend(environment({
      nvidiaDriverAvailable: true,
      cudaToolkitAvailable: true,
    }))).toBe("cuda")
  })

  it("fails clearly instead of silently building CPU on an NVIDIA machine without nvcc", () => {
    expect(() => selectLocalIcnBackend(environment({
      nvidiaDriverAvailable: true,
    }))).toThrow("nvcc is missing")
  })

  it("allows an explicit CPU build on an NVIDIA machine", () => {
    expect(selectLocalIcnBackend(environment({
      requested: "cpu",
      nvidiaDriverAvailable: true,
    }))).toBe("cpu")
  })

  it("selects the target-provided Metal backend on Apple Silicon", () => {
    expect(selectLocalIcnBackend(environment({
      platform: "darwin",
      arch: "arm64",
    }))).toBe("metal")
  })

  it("rejects incompatible explicit backends", () => {
    expect(() => selectLocalIcnBackend(environment({
      requested: "metal",
    }))).toThrow("requires Apple Silicon")
    expect(() => selectLocalIcnBackend(environment({
      requested: "cuda",
    }))).toThrow("no NVIDIA GPU")
  })
})
