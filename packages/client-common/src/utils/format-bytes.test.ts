import { describe, expect, it } from "vitest"
import {
  formatDecimalByteSize,
  formatDecimalGigabytes,
  formatDecimalMegabytesPerSecond,
} from "./format-bytes"

describe("decimal byte formatting", () => {
  it("formats memory in decimal gigabytes", () => {
    expect(formatDecimalGigabytes(3.4 * 1024 ** 3)).toBe("3.7 GB")
    expect(formatDecimalGigabytes(16 * 1024 ** 3)).toBe("17.2 GB")
  })

  it("rounds minimum requirements upward", () => {
    expect(formatDecimalGigabytes(2_000_000_001, { rounding: "up" })).toBe("2.1 GB")
  })

  it("formats transfer sizes with decimal units", () => {
    expect(formatDecimalByteSize(2.07 * 1024 ** 3)).toBe("2.22 GB")
    expect(formatDecimalByteSize(750_000_000)).toBe("750 MB")
  })

  it("formats transfer rates with decimal megabytes", () => {
    expect(formatDecimalMegabytesPerSecond(12_500_000)).toBe("13 MB/s")
  })
})
