import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { ModelRecipeRecommendation } from "@magnitudedev/icn"
import {
  ModelArtifactFingerprintSchema,
  ModelRecipeCatalogModelIdSchema,
  ModelRecipeConfigurationIdSchema,
} from "@magnitudedev/icn/provider"
import { candidateDetails } from "./local-model-inventory.js"

describe("ACN local recommendation projection", () => {
  it("preserves the ICN intent, explanation, and generation evidence without reranking", () => {
    const recommendation: ModelRecipeRecommendation = {
      configurationId: ModelRecipeConfigurationIdSchema.make("model:q6:p1:ctx200000"),
      catalogModelId: ModelRecipeCatalogModelIdSchema.make("model"),
      artifactFingerprint: ModelArtifactFingerprintSchema.make("owner/repo:commit:content"),
      modelId: Option.none(),
      intent: "fastest",
      displayName: "Model",
      family: "family",
      architecture: "moe",
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoningEfforts: [],
        defaultReasoningEffort: Option.none(),
      },
      totalParametersBillions: Option.some(35),
      activeParametersBillions: Option.some(3),
      effectiveParametersBillions: Option.none(),
      quantization: {
        format: "Q6_K",
        quantAwareCheckpoint: false,
        fidelityLabel: "Very high fidelity",
        fidelityEvidence: "Test evidence",
        fidelitySourceUrl: "https://example.com/fidelity",
      },
      quantTag: "Q6_K",
      repo: "owner/repo",
      revision: "commit",
      files: [],
      totalDownloadBytes: 10,
      sourcePageUrl: "https://example.com/model",
      license: { id: "test", url: "https://example.com/license", acknowledgementRequired: false },
      contextWindow: 200_000,
      estimatedRuntimeBytes: 20,
      stableCapacityBudgetBytes: 40,
      fitMarginBytes: 20,
      fitClass: "cpu_or_unified",
      constrainedContext: false,
      estimatedGeneration: Option.some({
        contextTokens: 200_000,
        lowerTokensPerSecond: 30,
        expectedTokensPerSecond: 35,
        upperTokensPerSecond: 40,
        confidence: "moderate",
        method: "estimator-v3",
      }),
      explanation: "Prioritizes responsive generation.",
    }

    const projected = Option.getOrThrow(candidateDetails(recommendation).recommendation)
    expect(projected).toMatchObject({
      intent: "fastest",
      explanation: "Prioritizes responsive generation.",
    })
    expect(Option.getOrThrow(projected.estimatedGeneration)).toMatchObject({
      contextTokens: 200_000,
      expectedTokensPerSecond: 35,
      method: "estimator-v3",
    })
  })
})
