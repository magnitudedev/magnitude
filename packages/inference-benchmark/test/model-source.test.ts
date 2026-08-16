import { describe, expect, it } from "vitest"
import { MODEL_PROFILES, parseHuggingFaceUrl, resolveModelReference } from "../src/model-source"

describe("model sources", () => {
  it("resolves a built-in immutable profile", () => {
    expect(resolveModelReference("qwen3.6-35b-a3b")).toMatchObject({
      kind: "huggingface",
      repository: "unsloth/Qwen3.6-35B-A3B-GGUF",
      revision: "a483e9e6cbd595906af30beda3187c2663a1118c",
      expectedSha256: "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450",
    })
    expect(MODEL_PROFILES.has("qwen3.6-35b-a3b")).toBe(true)
  })

  it("parses a Hugging Face resolve URL", () => {
    expect(parseHuggingFaceUrl("https://huggingface.co/owner/repo/resolve/main/models/model.gguf")).toEqual({
      kind: "huggingface",
      id: "model",
      repository: "owner/repo",
      revision: "main",
      file: "models/model.gguf",
    })
  })

  it("keeps a local path as an explicit override", () => {
    expect(resolveModelReference("served-name", "/models/model.gguf")).toEqual({
      kind: "local",
      id: "served-name",
      path: "/models/model.gguf",
    })
  })

  it("rejects an ambiguous bare model name", () => {
    expect(() => resolveModelReference("unknown-model")).toThrow(/model profile/)
  })
})
