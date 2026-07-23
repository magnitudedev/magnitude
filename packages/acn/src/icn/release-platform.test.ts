import { describe, expect, it } from "vitest"
import { selectIcnReleasePlatformKey } from "./release-platform"

const host = (
  overrides: Partial<Parameters<typeof selectIcnReleasePlatformKey>[0]> = {},
): Parameters<typeof selectIcnReleasePlatformKey>[0] => ({
  platform: "linux",
  arch: "arm64",
  requestedBackend: undefined,
  nvidiaDriverAvailable: false,
  ...overrides,
})

describe("ICN release platform selection", () => {
  it("selects the CUDA artifact for an NVIDIA Linux ARM64 host", () => {
    expect(selectIcnReleasePlatformKey(host({
      nvidiaDriverAvailable: true,
    }))).toBe("linux-arm64-cuda")
  })

  it("keeps the generic artifact for CPU-only Linux", () => {
    expect(selectIcnReleasePlatformKey(host())).toBe("linux-arm64")
  })

  it("allows an explicit CPU override on NVIDIA hardware", () => {
    expect(selectIcnReleasePlatformKey(host({
      requestedBackend: "cpu",
      nvidiaDriverAvailable: true,
    }))).toBe("linux-arm64")
  })

  it("rejects CUDA when no driver is available", () => {
    expect(() => selectIcnReleasePlatformKey(host({
      requestedBackend: "cuda",
    }))).toThrow("no NVIDIA driver")
  })
})
