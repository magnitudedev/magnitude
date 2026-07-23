import { describe, expect, it } from "vitest"
import { acceleratorDisplayName } from "./local-inference-hardware"

describe("local inference hardware projection", () => {
  it("uses native device descriptions instead of generic backend ordinals", () => {
    expect(acceleratorDisplayName({
      name: "CUDA0",
      description: "NVIDIA GB10",
    })).toBe("NVIDIA GB10")
    expect(acceleratorDisplayName({
      name: "MTL0",
      description: "Apple M4 Max",
    })).toBe("Apple M4 Max")
  })

  it("preserves a native device name that already identifies the hardware", () => {
    expect(acceleratorDisplayName({
      name: "NVIDIA RTX 5090",
      description: "NVIDIA RTX 5090",
    })).toBe("NVIDIA RTX 5090")
  })
})
