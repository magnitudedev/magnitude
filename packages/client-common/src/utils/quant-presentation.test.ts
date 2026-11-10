import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { formatQuantLabel, quantBitWidth } from "./quant-presentation"

describe("quant presentation", () => {
  it.each([
    ["UD-Q4_K_XL", 4],
    ["Q8_0", 8],
    ["IQ2_XXS", 2],
    ["TQ1_0", 1],
    ["MXFP4", 4],
    ["MXFP4_MOE", 4],
    ["BF16", 16],
    ["F16", 16],
    ["F32", 32],
    ["q6_k", 6],
  ])("reads the bit width of %s", (name, bits) => {
    expect(quantBitWidth(name)).toEqual(Option.some(bits))
  })

  it("has no bit width for a name that states none", () => {
    expect(quantBitWidth("model")).toEqual(Option.none())
    expect(quantBitWidth("SQUARE")).toEqual(Option.none())
  })

  it("labels a quant with its bit width and publisher", () => {
    expect(formatQuantLabel("UD-Q4_K_XL", Option.some("unsloth/Qwen3.6-35B-A3B-GGUF"))).toBe("4-bit unsloth UD-Q4_K_XL")
    expect(formatQuantLabel("custom", Option.none())).toBe("custom")
  })
})
