import { describe, expect, it } from "vitest"
import { windowsCapabilityLibrary, windowsImportedLibraries, windowsSystemLibrary } from "./windows-runtime"

describe("Windows runtime imports", () => {
  it("admits driver libraries only for the selected accelerator, never toolkit runtimes", () => {
    expect(windowsCapabilityLibrary("NVCUDA.dll", ["cuda"])).toBe(true)
    expect(windowsCapabilityLibrary("vulkan-1.dll", ["vulkan"])).toBe(true)
    expect(windowsCapabilityLibrary("nvcuda.dll", [])).toBe(false)
    expect(windowsCapabilityLibrary("vulkan-1.dll", ["cuda"])).toBe(false)
    expect(windowsCapabilityLibrary("nvcuda.dll", ["vulkan"])).toBe(false)
    for (const name of ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]) {
      expect(windowsCapabilityLibrary(name, ["cuda", "vulkan"])).toBe(false)
      expect(windowsSystemLibrary(name)).toBe(false)
    }
  })

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
