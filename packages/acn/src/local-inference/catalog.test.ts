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

  test("contains the corrected Qwen3.6 35B family and no stale large Qwen3.5 or GLM 5", () => {
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.modelId === "qwen3.6-35b-a3b")).toBe(true)
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.modelId.includes("qwen3.5-35"))).toBe(false)
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.displayName === "GLM 5")).toBe(false)
    expect(LOCAL_MODEL_CATALOG.some((entry) => entry.displayName === "GLM 5.2")).toBe(true)
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
    expect(LOCAL_MODEL_CATALOG.find((entry) => entry.id === "nemotron-3-super-120b-a12b:MXFP4_MOE")?.files).toHaveLength(3)
  })
})
