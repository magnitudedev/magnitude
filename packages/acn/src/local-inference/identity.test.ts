import { describe, expect, it } from "vitest"
import { LlamaCpp } from "@magnitudedev/local-inference"
import { providerModelIdForModelPath } from "./identity"

describe("llama.cpp provider model identity", () => {
  it("derives one opaque logical ID from the normalized model path", () => {
    const managed = LlamaCpp.normalizeLlamaModelPath("/models/./qwen.gguf")!
    const external = LlamaCpp.normalizeLlamaModelPath("/models/qwen.gguf")!
    expect(providerModelIdForModelPath(managed)).toBe(providerModelIdForModelPath(external))
    expect(providerModelIdForModelPath(managed)).toMatch(/^lmp_[a-f0-9]{64}$/)
  })

  it("does not merge equal filenames at different paths", () => {
    const first = LlamaCpp.normalizeLlamaModelPath("/a/model.gguf")!
    const second = LlamaCpp.normalizeLlamaModelPath("/b/model.gguf")!
    expect(providerModelIdForModelPath(first)).not.toBe(providerModelIdForModelPath(second))
  })
})
