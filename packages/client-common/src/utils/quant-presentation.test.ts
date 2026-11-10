import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { LocalModelFitsAssessment } from "@magnitudedev/sdk"
import { AssessmentEnvironmentIdSchema, ModelAssessmentIdSchema } from "@magnitudedev/sdk"
import { fitPlacementPhrase, formatQuantLabel, ladderBitWidths, quantBitWidth } from "./quant-presentation"

const GB = 1_000_000_000
const resident = (deviceBytes: number, hostBytes: number): LocalModelFitsAssessment => ({
  _tag: "Fits",
  assessmentId: ModelAssessmentIdSchema.make("assessment"),
  environmentId: AssessmentEnvironmentIdSchema.make("environment"),
  profile: { contextLength: 32_768 },
  served: { contextLength: 32_768 },
  memory: { domains: [], totalRequiredBytes: 0, requiredSystemMemoryBytes: 0, breakdown: { weightsBytes: 10 * GB, contextBytes: 2 * GB, overheadBytes: GB } },
  placement: { deviceBytes, hostBytes, diskBytes: 0 },
  residency: { _tag: "Resident", performance: { _tag: "Estimated", samples: [{ contextTokens: 32_768, estimatedTokensPerSecond: 40 }] } },
})

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

  it("gives the larger of two rungs sharing a bit width half a bit more", () => {
    expect(ladderBitWidths(["UD-IQ3_XXS", "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q6_K_XL"]))
      .toEqual([Option.some(3), Option.some(3.5), Option.some(4), Option.some(6)])
    expect(ladderBitWidths(["Q2_0", "IQ3_S", "custom"])).toEqual([Option.some(2), Option.some(3), Option.none()])
  })

  it("says a model holding every weight fits the memory it serves from", () => {
    expect(fitPlacementPhrase(resident(10 * GB, 0), "Device")).toBe("Fits in GPU")
    expect(fitPlacementPhrase(resident(10 * GB, 0), "Unified")).toBe("Fits in memory")
    expect(fitPlacementPhrase(resident(10 * GB, 0), "System")).toBe("Fits in RAM")
  })
})
