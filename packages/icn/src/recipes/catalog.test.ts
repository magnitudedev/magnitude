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
      reviewedAt: "2026-07-22",
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

  it("uses one comparable Terminal-Bench v2.1 capability cohort", () => {
    const scores = Object.fromEntries(MODEL_RECIPE_REGISTRY.models.map((model) => {
      const evidence = required(
        Option.fromNullable(model.performance.benchmarks.find(({ benchmarkId }) =>
          benchmarkId === "terminal-bench-v2.1")),
        `${model.id} Terminal-Bench evidence`,
      )
      return [model.id, evidence.score]
    }))
    expect(scores).toEqual({
      "qwen3.5-4b": 25.8,
      "qwen3.5-9b": 29.2,
      "qwen3.6-27b": 60.7,
      "qwen3.6-35b-a3b": 44.9,
      "gemma-4-e2b-it-qat": 15,
      "gemma-4-12b-it-qat": 21,
      "gemma-4-26b-a4b-it-qat": 39,
      "gemma-4-31b-it-qat": 43.4,
      "qwen3.5-122b-a10b": 47.6,
      "nemotron-3-super-120b-a12b": 38.6,
      "deepseek-v4-flash": 61.8,
      "nemotron-3-ultra-550b-a55b": 53.9,
      "glm-5.2": 77.9,
    })
  })

  it("marks only the two unmeasured Gemma scores as explicit estimates", () => {
    const estimated = MODEL_RECIPE_REGISTRY.models.flatMap((model) =>
      model.performance.benchmarks
        .filter(({ provenance }) => provenance === "estimated_terminal_bench_2.1")
        .map((evidence) => ({ id: model.id, score: evidence.score, basis: evidence.basis })))
    expect(estimated).toEqual([
      expect.objectContaining({ id: "gemma-4-e2b-it-qat", score: 15 }),
      expect.objectContaining({ id: "gemma-4-12b-it-qat", score: 21 }),
    ])
    expect(estimated.every(({ basis }) => basis.length > 0)).toBe(true)
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

  it("rejects duplicate or malformed Terminal-Bench evidence", () => {
    const original = required(Option.fromNullable(MODEL_RECIPE_REGISTRY.models.at(0)), "first model recipe")
    const evidence = required(Option.fromNullable(original.performance.benchmarks.at(0)), "capability evidence")
    const broken: ModelRecipeRegistry = {
      ...MODEL_RECIPE_REGISTRY,
      models: [{
        ...original,
        performance: {
          ...original.performance,
          benchmarks: [evidence, {
            ...evidence,
            score: 101,
            provenance: "estimated_terminal_bench_2.1",
            basis: "",
          }],
        },
      }],
    }
    const issues = validateModelRecipeRegistry(broken)
    expect(issues).toContain(`${original.id} must have exactly one Terminal-Bench v2.1 capability score`)
    expect(issues).toContain(`${original.id} has invalid benchmark evidence`)
    expect(issues).toContain(`${original.id} has benchmark evidence without a stated basis`)
  })
})
