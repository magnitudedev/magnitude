import { describe, expect, it } from "vitest"
import {
  ModelRecipeConfigurationIdSchema,
  NativeIcnModelIdSchema,
  candidateLocalModelId,
  localProviderModelId,
  nativeLocalModelId,
} from "./model-identity"

describe("private-to-product local model identity", () => {
  it("is deterministic, namespaced, and does not expose native identity", () => {
    const nativeId = NativeIcnModelIdSchema.make("/private/cache/model.gguf")
    const native = nativeLocalModelId(nativeId)
    expect(native).toBe(nativeLocalModelId(nativeId))
    expect(native).toMatch(/^native_[a-f0-9]{32}$/)
    expect(native).not.toContain("private")
    expect(localProviderModelId(native)).toBe(`local:${native}`)
  })

  it("keeps candidate and native identity domains distinct", () => {
    const candidate = candidateLocalModelId({
      configurationId: ModelRecipeConfigurationIdSchema.make("recipe:model:ctx"),
    })
    const native = nativeLocalModelId(NativeIcnModelIdSchema.make("recipe:model:ctx"))
    expect(candidate).toMatch(/^candidate_[a-f0-9]{32}$/)
    expect(candidate).not.toBe(native)
  })
})
