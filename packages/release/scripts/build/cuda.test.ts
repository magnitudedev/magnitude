import { describe, expect, it } from "vitest"
import { inspectNvccCompiler, inspectPtxImages } from "./cuda"

describe("CUDA artifact inspection", () => {
  it("extracts the compiler identity rather than the trailing build label", () => {
    expect(inspectNvccCompiler(`nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0`)).toBe(
      "Cuda compilation tools, release 12.9, V12.9.86",
    )
  })

  it("derives ordinary and architecture-specific images from dumped PTX", () => {
    expect(inspectPtxImages(`
.version 8.8
.target sm_80
.address_size 64
.version 8.8
.target sm_120a
.address_size 64
.version 8.8
.target sm_80
`)).toEqual([
      { ptxVersion: "8.8", target: 80, architectureSpecific: false },
      { ptxVersion: "8.8", target: 120, architectureSpecific: true },
    ])
  })

  it("ignores cuobjdump framing that is not PTX", () => {
    expect(inspectPtxImages("Fatbin elf code:\narch = sm_80")).toEqual([])
  })
})
