import { describe, expect, it } from "vitest"
import { windowsImportedLibraries, windowsSystemLibrary } from "./windows-runtime"

describe("Windows runtime imports", () => {
  it("reads dependency names without treating dumpbin headings or input paths as imports", () => {
    expect(windowsImportedLibraries(`
Dump of file C:\\build\\ggml-base.dll
  Image has the following dependencies:
    KERNEL32.dll
    MSVCP140.dll
    VCRUNTIME140_1.dll
    api-ms-win-crt-runtime-l1-1-0.dll
  Image has the following delay load dependencies:
    MSVCP140.dll
  Summary
    1000 .data
`)).toEqual(["kernel32.dll", "msvcp140.dll", "vcruntime140_1.dll", "api-ms-win-crt-runtime-l1-1-0.dll"])
  })

  it("requires app-local compiler runtimes and never treats arbitrary paths as OS libraries", () => {
    for (const name of ["KERNEL32.dll", "api-ms-win-crt-heap-l1-1-0.dll"]) expect(windowsSystemLibrary(name)).toBe(true)
    for (const name of ["MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll", "ggml-base.dll", "C:\\Windows\\KERNEL32.dll", "api-ms-win-../vendor.dll"]) {
      expect(windowsSystemLibrary(name)).toBe(false)
    }
  })
})
