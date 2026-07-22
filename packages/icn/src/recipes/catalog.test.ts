import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  MODEL_RECIPE_REGISTRY,
  resolveModelRecipeArtifact,
  validateModelRecipeRegistry,
} from "./catalog"
import type { ModelRecipeRegistry } from "./types"

describe("canonical local model catalog overlay", () => {
  const required = <A>(value: Option.Option<A>, description: string): A =>
    Option.getOrThrowWith(value, () => new Error(`Missing ${description}`))

  it("is internally valid and records its evidence review date", () => {
    expect(MODEL_RECIPE_REGISTRY).toMatchObject({
      reviewedAt: "2026-07-20",
    })
    expect(validateModelRecipeRegistry(MODEL_RECIPE_REGISTRY)).toEqual([])
  })

  it("contains stable identities and selectors, never Hub snapshots", () => {
    const encoded = JSON.stringify(MODEL_RECIPE_REGISTRY)
    expect(encoded).not.toContain('"revision"')
    expect(encoded).not.toContain('"sha256"')
    expect(encoded).not.toContain('"sizeBytes"')
    expect(encoded).not.toContain('"primaryFile"')
    expect(MODEL_RECIPE_REGISTRY.models).toHaveLength(13)
    expect(MODEL_RECIPE_REGISTRY.models.flatMap(({ artifacts }) => artifacts)).toHaveLength(25)
  })

  it("groups fidelity choices under one canonical model", () => {
    const model = required(Option.fromNullable(MODEL_RECIPE_REGISTRY.models.find(({ id }) => id === "qwen3.6-35b-a3b")), "Qwen recipe")
    expect(model.artifacts.map(({ quantization }) => quantization.format)).toEqual([
      "UD-Q4_K_XL",
      "UD-Q5_K_XL",
      "UD-Q6_K_XL",
      "UD-Q8_K_XL",
    ])
  })

  it("resolves an unsharded selector against a live snapshot", () => {
    const model = required(Option.fromNullable(MODEL_RECIPE_REGISTRY.models.at(0)), "first model recipe")
    const candidate = required(Option.fromNullable(model.artifacts.at(0)), "first model artifact")
    const resolved = resolveModelRecipeArtifact(model, candidate, {
      repository: candidate.repository,
      commit: "a".repeat(40),
      license: Option.some("apache-2.0"),
      licenseUrl: Option.none(),
      ggufFiles: [
        { path: "model-UD-Q4_K_XL.gguf", sizeBytes: Option.some(100) },
        { path: "model-UD-Q5_K_XL.gguf", sizeBytes: Option.some(120) },
        { path: "mmproj-model-UD-Q4_K_XL.gguf", sizeBytes: Option.some(5) },
      ],
    })
    expect(Option.getOrThrow(resolved)).toMatchObject({
      revision: "a".repeat(40),
      primaryGguf: "model-UD-Q4_K_XL.gguf",
      publishedWeightBytes: 100,
      quantTag: "UD-Q4_K_XL",
    })
  })

  it("selects only the first file of a complete shard family as preview primary", () => {
    const model = required(Option.fromNullable(MODEL_RECIPE_REGISTRY.models.find(({ id }) => id === "glm-5.2")), "GLM recipe")
    const candidate = required(Option.fromNullable(model.artifacts.at(0)), "GLM artifact")
    const resolved = resolveModelRecipeArtifact(model, candidate, {
      repository: candidate.repository,
      commit: "b".repeat(40),
      license: Option.none(),
      licenseUrl: Option.none(),
      ggufFiles: [
        { path: "UD-Q4_K_XL/model-UD-Q4_K_XL-00002-of-00003.gguf", sizeBytes: Option.some(20) },
        { path: "UD-Q4_K_XL/model-UD-Q4_K_XL-00001-of-00003.gguf", sizeBytes: Option.some(10) },
        { path: "UD-Q4_K_XL/model-UD-Q4_K_XL-00003-of-00003.gguf", sizeBytes: Option.some(30) },
      ],
    })
    expect(Option.getOrThrow(resolved)).toMatchObject({
      primaryGguf: "UD-Q4_K_XL/model-UD-Q4_K_XL-00001-of-00003.gguf",
      publishedWeightBytes: 60,
    })
  })

  it("refuses ambiguous or missing selectors", () => {
    const model = required(Option.fromNullable(MODEL_RECIPE_REGISTRY.models.at(0)), "first model recipe")
    const candidate = required(Option.fromNullable(model.artifacts.at(0)), "first model artifact")
    const snapshot = {
      repository: candidate.repository,
      commit: "c".repeat(40),
      license: Option.none<string>(),
      licenseUrl: Option.none<string>(),
      ggufFiles: [],
    }
    expect(Option.isNone(resolveModelRecipeArtifact(model, candidate, snapshot))).toBe(true)
    expect(Option.isNone(resolveModelRecipeArtifact(model, candidate, {
      ...snapshot,
      ggufFiles: [
        { path: "a-UD-Q4_K_XL.gguf", sizeBytes: Option.some(10) },
        { path: "b-UD-Q4_K_XL.gguf", sizeBytes: Option.some(10) },
      ],
    }))).toBe(true)
  })

  it("rejects duplicate IDs and pinned-path selectors", () => {
    const original = required(Option.fromNullable(MODEL_RECIPE_REGISTRY.models.at(0)), "first model recipe")
    const originalArtifact = required(Option.fromNullable(original.artifacts.at(0)), "first model artifact")
    const broken: ModelRecipeRegistry = {
      ...MODEL_RECIPE_REGISTRY,
      models: [original, {
        ...original,
        artifacts: [{ ...originalArtifact, filenameIncludes: "model.gguf" }],
      }],
    }
    const issues = validateModelRecipeRegistry(broken)
    expect(issues).toContain(`duplicate model id ${original.id}`)
    expect(issues).toContain(`duplicate artifact id ${originalArtifact.id}`)
    expect(issues.some((issue) => issue.includes("pinned path"))).toBe(true)
  })
})
