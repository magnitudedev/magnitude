import { describe, expect, test } from "vitest"
import { LOCAL_MODEL_CATALOG, catalogFileUrl, catalogSourcePageUrl } from "./catalog"

describe("local model catalog", () => {
  test("pins every shipping artifact to exact immutable metadata", () => {
    const ids = new Set<string>()
    for (const entry of LOCAL_MODEL_CATALOG) {
      expect(ids.has(entry.id), `duplicate catalog id ${entry.id}`).toBe(false)
      ids.add(entry.id)
      expect(entry.revision).toMatch(/^[0-9a-f]{40}$/)
      expect(entry.files.length).toBeGreaterThan(0)
      expect(entry.license.id).toBeTruthy()
      expect(entry.quantization.fidelityLabel).not.toBe("Accuracy")
      expect(entry.quantization.fidelityLabel).toMatch(/^(Good|High|Very high|Near-original) fidelity /)
      expect(entry.quantization.fidelityLabel).not.toContain("—")
      expect(entry.quantization.fidelityLabel).not.toMatch(/Unsloth|NVIDIA|Google/i)
      expect(entry.quantization.fidelityLabel.split(/\s+/).length).toBeGreaterThanOrEqual(5)
      expect(entry.quantization.fidelityLabel.split(/\s+/).length).toBeLessThanOrEqual(10)
      expect(entry.quantization.fidelityEvidence.length).toBeGreaterThan(20)
      for (const file of entry.files) {
        expect(file.sizeBytes).toBeGreaterThan(0)
        expect(file.sha256).toMatch(/^[0-9a-f]{64}$/)
        expect(catalogFileUrl(entry, file.path)).toBe(
          `https://huggingface.co/${entry.repo}/resolve/${entry.revision}/${file.path}`,
        )
      }
      expect(catalogSourcePageUrl(entry)).toContain(`/blob/${entry.revision}/`)
    }
  })

  test("contains the current Qwen and Gemma families without invented model sizes", () => {
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.modelId === "qwen3.6-35b-a3b")).toBe(true)
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.modelId.includes("qwen3.5-35"))).toBe(false)
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.family === "qwen3")).toBe(false)
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.displayName === "Qwen3.5 12B")).toBe(false)
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.displayName === "Gemma 4 E2B")).toBe(true)
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.displayName === "Gemma 4 12B")).toBe(true)
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.displayName === "GLM 5")).toBe(false)
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.displayName === "GLM 5.2 753B-A40B")).toBe(true)
  })

  test("pins the reviewed representative artifacts", () => {
    expect(LOCAL_MODEL_CATALOG.find((entry) => entry.id === "qwen3.6-27b:UD-Q5_K_XL")).toMatchObject({
      revision: "82d411acf4a06cfb8d9b073a5211bf410bfc29bf",
      files: [{
        path: "Qwen3.6-27B-UD-Q5_K_XL.gguf",
        sizeBytes: 20_038_256_864,
        sha256: "ac310abf2895aa397121bad6c0be89466af41f0f1606a21c1131b110eeb19d0e",
      }],
    })
    expect(LOCAL_MODEL_CATALOG.find((entry) => entry.id === "qwen3.6-35b-a3b:UD-Q5_K_XL")).toMatchObject({
      files: [{ sizeBytes: 26_592_508_896 }],
      activeParametersBillions: 3,
    })
    expect(LOCAL_MODEL_CATALOG.find((entry) => entry.id === "gemma-4-26b-a4b-it-qat:UD-Q4_K_XL")?.quantization).toMatchObject({
      quantAwareCheckpoint: true,
      bitsClass: "q4",
    })
    expect(LOCAL_MODEL_CATALOG.find((entry) => entry.id === "gemma-4-e2b-it-qat:UD-Q4_K_XL")).toMatchObject({
      displayName: "Gemma 4 E2B",
      totalParametersBillions: 5.1,
      effectiveParametersBillions: 2.3,
      modelMaximumContextTokens: 131_072,
      files: [{ sizeBytes: 2_620_368_960 }],
    })
    expect(LOCAL_MODEL_CATALOG.find((entry) => entry.id === "gemma-4-12b-it-qat:UD-Q4_K_XL")).toMatchObject({
      displayName: "Gemma 4 12B",
      totalParametersBillions: 11.95,
      modelMaximumContextTokens: 262_144,
      files: [{ sizeBytes: 6_716_355_328 }],
    })
    expect(LOCAL_MODEL_CATALOG.find((entry) => entry.id === "nemotron-3-super-120b-a12b:MXFP4_MOE")?.files).toHaveLength(3)
  })

  test("includes the restored workstation and very-large capacity tiers", () => {
    expect(LOCAL_MODEL_CATALOG.find((entry) => entry.id === "qwen3.5-122b-a10b:UD-Q4_K_XL")).toMatchObject({
      totalParametersBillions: 122,
      activeParametersBillions: 10,
      files: expect.arrayContaining([expect.objectContaining({ sizeBytes: 49_640_779_424 })]),
    })
    expect(LOCAL_MODEL_CATALOG.find((entry) => entry.id === "deepseek-v4-flash:UD-Q8_K_XL")).toMatchObject({
      displayName: "DeepSeek V4 Flash 284B-A13B",
      totalParametersBillions: 284,
      activeParametersBillions: 13,
      quantization: { bitsClass: "q8" },
    })
    expect(LOCAL_MODEL_CATALOG.find((entry) => entry.id === "nemotron-3-ultra-550b-a55b:MXFP4_MOE")).toMatchObject({
      totalParametersBillions: 550,
      activeParametersBillions: 55,
      files: expect.arrayContaining([expect.objectContaining({ sizeBytes: 47_459_816_960 })]),
    })
  })

  test("does not present cross-model quant guidance as an exact artifact measurement", () => {
    const qwen27Q5 = LOCAL_MODEL_CATALOG.find((entry) => entry.id === "qwen3.6-27b:UD-Q5_K_XL")
    expect(qwen27Q5?.quantization).toMatchObject({
      fidelityLabel: "High fidelity with only minor quality loss",
      fidelitySourceUrl: "https://arxiv.org/abs/2606.19558",
    })
    expect(qwen27Q5?.quantization.fidelityEvidence).toContain("Cross-model guidance")
    expect(qwen27Q5?.quantization.fidelityEvidence).not.toContain("0.045526")
    expect(qwen27Q5?.quantization.fidelityEvidence).not.toContain("96.187")
  })

  test("records exact Qwen3.6 35B-A3B Q4 and Q5 measurements without treating KLD as accuracy", () => {
    const q4 = LOCAL_MODEL_CATALOG.find((entry) => entry.id === "qwen3.6-35b-a3b:UD-Q4_K_XL")
    const q5 = LOCAL_MODEL_CATALOG.find((entry) => entry.id === "qwen3.6-35b-a3b:UD-Q5_K_XL")
    const q6 = LOCAL_MODEL_CATALOG.find((entry) => entry.id === "qwen3.6-35b-a3b:UD-Q6_K_XL")

    expect(q4?.quantization.fidelityLabel).toBe("Good fidelity in model-specific testing")
    expect(q4?.quantization.fidelityRank).toBe(40)
    expect(q4?.quantization.fidelityEvidence).toContain("KLD 0.0135")
    expect(q4?.quantization.fidelityEvidence).toContain("composite benchmark score 0.728")
    expect(q5?.quantization.fidelityLabel).toBe("High fidelity in model-specific testing")
    expect(q5?.quantization.fidelityRank).toBe(50)
    expect(q5?.quantization.fidelityEvidence).toContain("KLD 0.0082")
    expect(q5?.quantization.fidelityEvidence).toContain("did not produce a higher composite score")
    expect(q6?.quantization.fidelityLabel).toBe("Very high fidelity with minimal quality loss")
    expect(q6?.quantization.fidelityEvidence).toContain("Cross-model guidance")
  })

  test("uses narrow evidence-backed claims for QAT, FP4-trained, and GLM artifacts", () => {
    const gemma = LOCAL_MODEL_CATALOG.find((entry) => entry.id === "gemma-4-26b-a4b-it-qat:UD-Q4_K_XL")
    const nemotron = LOCAL_MODEL_CATALOG.find((entry) => entry.id === "nemotron-3-super-120b-a12b:MXFP4_MOE")
    const glm = LOCAL_MODEL_CATALOG.find((entry) => entry.id === "glm-5.2:UD-Q4_K_XL")

    expect(gemma?.quantization.fidelityLabel).toBe("Near-original fidelity from quantization-aware training")
    expect(gemma?.quantization.fidelityEvidence).toContain("no KLD or downstream delta is published")
    expect(nemotron?.quantization.fidelityLabel).toBe("Near-original fidelity in benchmark comparisons")
    expect(nemotron?.quantization.fidelityEvidence).toContain("78.44 versus 78.69")
    expect(nemotron?.quantization.fidelityEvidence).toContain("separate conversion")
    expect(glm?.quantization.fidelityLabel).toBe("Near-original fidelity in quantization tests")
    expect(glm?.totalParametersBillions).toBe(753)
    expect(glm?.activeParametersBillions).toBe(40)
    expect(glm?.quantization.fidelityRank).toBe(40)
    expect(glm?.quantization.fidelityEvidence).toContain("Q8_0 defined as 100%")
    expect(glm?.quantization.fidelityEvidence).toContain("not BF16 fidelity or coding accuracy")
  })
})
