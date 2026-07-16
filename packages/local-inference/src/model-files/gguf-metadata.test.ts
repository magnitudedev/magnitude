import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { GgufMetadata, normalizeParameterCount, projectGgufMetadata } from "./gguf-metadata"

const value = (entry: unknown) => ({ value: entry })

describe("GGUF metadata projection", () => {
  it("projects authoritative typed metadata without inventing absent fields", () => {
    const metadata = new GgufMetadata({
      "general.name": value("Exact model name"),
      "general.architecture": value("qwen2"),
      "general.file_type": value(15),
      "qwen2.context_length": value(131_072n),
      "qwen2.embedding_length": value(8192),
      "tokenizer.ggml.model": value("gpt2"),
      "general.base_model.0.name": value("Base model"),
      "general.base_model.0.repo": value("owner/base"),
    })

    const projected = projectGgufMetadata(metadata, normalizeParameterCount(Option.some(32_000_000_000n)))

    expect(Option.getOrNull(projected.name)).toBe("Exact model name")
    expect(Option.getOrNull(projected.architecture)).toBe("qwen2")
    expect(Option.getOrNull(projected.ggufFileType)).toBe(15)
    expect(Option.getOrNull(projected.trainedContextTokens)).toBe(131_072)
    expect(Option.getOrNull(projected.embeddingLength)).toBe(8192)
    expect(Option.getOrNull(projected.parameterCount)).toBe(32_000_000_000)
    expect(projected.baseModelNames).toEqual(["Base model"])
    expect(projected.baseModelRepositories).toEqual(["owner/base"])
    expect(Option.isNone(projected.blockCount)).toBe(true)
    expect(Option.isNone(projected.inputModalities)).toBe(true)
  })

  it("rejects values whose typed metadata value has the wrong type", () => {
    const projected = projectGgufMetadata(new GgufMetadata({
      "general.name": value(42),
      "general.file_type": value("15"),
      "general.base_model.0.name": value("First"),
      "general.base_model.1.name": value(2),
    }), Option.none())

    expect(Option.isNone(projected.name)).toBe(true)
    expect(Option.isNone(projected.ggufFileType)).toBe(true)
    expect(projected.baseModelNames).toEqual(["First"])
  })
})
