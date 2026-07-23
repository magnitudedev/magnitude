import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { ModelRecipesState } from "./schema.js"
import { ReasoningEffortSchema } from "@magnitudedev/ai"
import {
  ModelArtifactFingerprintSchema,
  ModelRecipeCatalogModelIdSchema,
  ModelRecipeConfigurationIdSchema,
} from "../provider/model-identity.js"

describe("ModelRecipesState wire schema", () => {
  it("encodes Option values as JSON-safe nullable fields", () => {
    const recommendation = {
      configurationId: ModelRecipeConfigurationIdSchema.make("configuration"),
      catalogModelId: ModelRecipeCatalogModelIdSchema.make("model"),
      artifactFingerprint: ModelArtifactFingerprintSchema.make("owner/repo:commit:content"),
      modelId: Option.none(),
      intent: "balanced" as const,
      displayName: "Model",
      family: "family",
      architecture: "moe" as const,
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoningEfforts: [ReasoningEffortSchema.make("high")],
        defaultReasoningEffort: Option.some(ReasoningEffortSchema.make("high")),
      },
      totalParametersBillions: Option.some(35),
      activeParametersBillions: Option.some(3),
      effectiveParametersBillions: Option.none<number>(),
      quantization: {
        format: "Q4_K_M",
        quantAwareCheckpoint: false,
        fidelityLabel: "Quantized",
        fidelityEvidence: "Test fixture",
        fidelitySourceUrl: "https://example.com",
      },
      quantTag: "Q4_K_M",
      repo: "owner/repo",
      revision: "commit",
      files: [],
      totalDownloadBytes: 1,
      sourcePageUrl: "https://example.com",
      license: { id: "test", url: "https://example.com", acknowledgementRequired: false },
      contextWindow: 1,
      estimatedRuntimeBytes: 1,
      stableCapacityBudgetBytes: 1,
      fitMarginBytes: 0,
      fitClass: "full_accelerator" as const,
      constrainedContext: false,
      estimatedGeneration: Option.some({
        contextTokens: 1,
        lowerTokensPerSecond: 10,
        expectedTokensPerSecond: 12,
        upperTokensPerSecond: 14,
        confidence: "high" as const,
        method: "test-estimator",
      }),
      explanation: "Test fixture",
    }
    const state = { _tag: "Ready" as const, recommendations: [recommendation], failureCount: 0 }

    const encoded = Schema.encodeSync(ModelRecipesState)(state)
    expect(encoded._tag).toBe("Ready")
    if (encoded._tag !== "Ready") throw new Error("Expected encoded recipe state to be ready")
    expect(encoded.recommendations[0]).toMatchObject({
      totalParametersBillions: 35,
      activeParametersBillions: 3,
      estimatedGeneration: {
        expectedTokensPerSecond: 12,
      },
    })
    expect(encoded.recommendations[0]).not.toHaveProperty("effectiveParametersBillions")

    const decoded = Schema.decodeUnknownSync(ModelRecipesState)(encoded)
    expect(decoded._tag).toBe("Ready")
    if (decoded._tag !== "Ready") throw new Error("Expected decoded recipe state to be ready")
    expect(Option.getOrThrow(decoded.recommendations[0].totalParametersBillions)).toBe(35)
    expect(Option.isNone(decoded.recommendations[0].effectiveParametersBillions)).toBe(true)
  })
})
