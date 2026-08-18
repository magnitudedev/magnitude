import { describe, expect, it } from "vitest"
import {
  formatMemorySize,
  formatStorageSize,
  formatTransferRate,
} from "./format-bytes"

describe("byte formatting", () => {
  it("formats memory using the hardware convention", () => {
    expect(formatMemorySize(3.4 * 1024 ** 3)).toBe("3.4 GB")
    expect(formatMemorySize(16 * 1024 ** 3)).toBe("16 GB")
    expect(formatMemorySize(512 * 1024 ** 3)).toBe("512 GB")
    expect(formatMemorySize(768 * 1024 ** 2)).toBe("768 MB")
    expect(formatMemorySize(2 * 1024 ** 4)).toBe("2 TB")
  })

  it("rounds minimum requirements upward", () => {
    expect(formatMemorySize(2 * 1024 ** 3 + 1, { rounding: "up" })).toBe("2.1 GB")
  })

  it("formats storage sizes with adaptive decimal units", () => {
    expect(formatStorageSize(500)).toBe("500 B")
    expect(formatStorageSize(100_000)).toBe("100 KB")
    expect(formatStorageSize(12_500_000)).toBe("12.5 MB")
    expect(formatStorageSize(2.07 * 1024 ** 3)).toBe("2.22 GB")
    expect(formatStorageSize(750_000_000)).toBe("750 MB")
    expect(formatStorageSize(2_000_000_000_000)).toBe("2 TB")
    expect(formatStorageSize(3_000_000_000_000_000)).toBe("3 PB")
  })

  it("promotes values that round across a unit boundary", () => {
    expect(formatStorageSize(999_999)).toBe("1 MB")
    expect(formatMemorySize(1024 ** 3 - 1, { rounding: "up" })).toBe("1 GB")
  })

  it("formats transfer rates with decimal megabytes", () => {
    expect(formatTransferRate(12_500_000)).toBe("13 MB/s")
  })
})
