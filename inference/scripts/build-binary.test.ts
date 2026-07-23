import { describe, expect, it } from "vitest"
import { icnReleasePlatformKey } from "./build-binary"

describe("ICN release artifact naming", () => {
  it("keeps the generic CPU platform key", () => {
    expect(icnReleasePlatformKey("linux-arm64", "cpu")).toBe("linux-arm64")
  })

  it("gives CUDA artifacts an independent platform key", () => {
    expect(icnReleasePlatformKey("linux-arm64", "cuda")).toBe("linux-arm64-cuda")
  })
})
