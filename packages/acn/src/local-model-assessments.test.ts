import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"

import {
  formatLocalModelAssessmentFailure,
  localModelAssessmentProfiles,
  localModelAssessmentResultFromIcn,
  performanceSampleContextTokens,
} from "./local-model-assessments"
import {
  AssessmentEnvironmentIdSchema,
  type ServableModelBundle,
} from "@magnitudedev/acn-protocol"
import type { ModelServingConfiguration as NativeModelServingConfiguration } from "@magnitudedev/icn-protocol/schemas"

const standaloneBundle = (maximumContextLength: number): ServableModelBundle => ({
  _tag: "Standalone",
  package: { properties: { maximumContextLength } },
} as unknown as ServableModelBundle)

describe("localModelAssessmentProfiles", () => {
  it("defaults discovered local models to the 100K baseline", () => {
    expect(localModelAssessmentProfiles(standaloneBundle(131_072))).toEqual([
      { contextLength: 100_000 },
    ])
    expect(localModelAssessmentProfiles(standaloneBundle(300_000))).toEqual([
      { contextLength: 100_000 },
    ])
  })

  it("uses the catalog context when provided", () => {
    expect(localModelAssessmentProfiles(standaloneBundle(131_072), 50_000)).toEqual([
      { contextLength: 50_000 },
    ])
  })

  it("bounds a catalog context by the bundle maximum", () => {
    expect(localModelAssessmentProfiles(standaloneBundle(40_000), 50_000)).toEqual([
      { contextLength: 40_000 },
    ])
  })

  it("uses the model maximum when it is below the local baseline", () => {
    expect(localModelAssessmentProfiles(standaloneBundle(80_000))).toEqual([
      { contextLength: 80_000 },
    ])
  })

  it("uses the lower package limit for speculative decoding", () => {
    const target = {
      _tag: "SpeculativeDecodingPair",
      target: { properties: { maximumContextLength: 131_072 } },
      draft: { properties: { maximumContextLength: 32_768 } },
    } as unknown as ServableModelBundle
    expect(localModelAssessmentProfiles(target)).toEqual([
      { contextLength: 32_768 },
    ])
  })

  it("caps a speculative pair when both package limits exceed 100K", () => {
    const target = {
      _tag: "SpeculativeDecodingPair",
      target: { properties: { maximumContextLength: 262_144 } },
      draft: { properties: { maximumContextLength: 131_072 } },
    } as unknown as ServableModelBundle
    expect(localModelAssessmentProfiles(target)).toEqual([
      { contextLength: 100_000 },
    ])
  })

  it("does not invent a profile below the product minimum", () => {
    expect(localModelAssessmentProfiles(standaloneBundle(2_048))).toEqual([])
  })
})

describe("performanceSampleContextTokens", () => {
  it("samples 25K, 50K, 75K, and the full configured context", () => {
    expect(performanceSampleContextTokens({ contextLength: 100_000 })).toEqual([
      25_000,
      50_000,
      75_000,
      100_000,
    ])
  })

  it("bounds and deduplicates samples for shorter-context models", () => {
    expect(performanceSampleContextTokens({ contextLength: 32_768 })).toEqual([
      25_000,
      32_768,
    ])
    expect(performanceSampleContextTokens({ contextLength: 20_000 })).toEqual([20_000])
  })
})

describe("localModelAssessmentResultFromIcn", () => {
  const environmentId = AssessmentEnvironmentIdSchema.make("environment-test")
  const assessmentBundle = {
    _tag: "Standalone" as const,
    package: {
      id: "package-test",
      source: { _tag: "Local" as const, path: "/models/test.gguf" },
      files: [{
        id: "file-test",
        path: "test.gguf",
        role: "weights" as const,
        sizeBytes: 1,
        tensorStorageBytes: Option.none<number>(),
        sha256: "a".repeat(64),
      }],
      relationships: [],
      properties: {
        format: "gguf",
        quantization: "Q4_K_M",
        quantizationName: "4-bit",
        architecture: "test",
        maximumContextLength: 100_000,
      },
    },
  }
  const nativeConfiguration = (
    id: string,
    contextLength: number,
  ): NativeModelServingConfiguration => ({
    id,
    bundle: assessmentBundle,
    profile: { contextLength },
  }) as unknown as NativeModelServingConfiguration

  it("preserves terminal non-capacity assessment evidence", () => {
    const result = Effect.runSync(localModelAssessmentResultFromIcn({
      _tag: "Assessed",
      requestId: "assessment-0",
      profiles: [{
        _tag: "DoesNotFit",
        configuration: nativeConfiguration("configuration-0", 50_000),
        assessmentId: "assessment-result-0",
        memory: [],
        totalRequiredBytes: 0,
        limitingResource: "system_memory",
        deficitBytes: 1024,
      }, {
        _tag: "Incompatible",
        configuration: nativeConfiguration("configuration-1", 100_000),
        failure: {
          code: "unsupported_architecture",
          message: "Unsupported architecture",
          retryable: false,
        },
      }],
    }, environmentId))

    expect(result).toEqual({
      _tag: "Assessed",
      environmentId,
      assessments: [{
        _tag: "DoesNotFit",
        configuration: {
          id: "configuration-0",
          bundle: assessmentBundle,
          profile: { contextLength: 50_000 },
        },
        assessmentId: "assessment-result-0",
        memory: [],
        totalRequiredBytes: 0,
        limitingResource: "system_memory",
        deficitBytes: 1024,
      }, {
        _tag: "Incompatible",
        configuration: {
          id: "configuration-1",
          bundle: assessmentBundle,
          profile: { contextLength: 100_000 },
        },
        failure: {
          code: "unsupported_architecture",
          message: "Unsupported architecture",
          retryable: false,
        },
      }],
    })
  })
})

describe("formatLocalModelAssessmentFailure", () => {
  it("preserves structured remote error details for internal diagnostics", () => {
    const detail = formatLocalModelAssessmentFailure({
      _tag: "GeneratedClientRemoteError",
      operationId: "assessModels",
      status: 500,
      body: {
        error: {
          code: "inventory_error",
          message: "isolated native planner exited with status 1",
          retryable: true,
          type: "server_error",
        },
      },
    })

    expect(detail).toContain("GeneratedClientRemoteError")
    expect(detail).toContain("assessModels")
    expect(detail).toContain("inventory_error")
    expect(detail).toContain("isolated native planner exited with status 1")
  })

  it("preserves ordinary error stacks", () => {
    const detail = formatLocalModelAssessmentFailure(new Error("hardware assessment failed"))

    expect(detail).toContain("Error: hardware assessment failed")
    expect(detail).toContain("local-model-assessments.test.ts")
  })

  it("preserves nested error causes", () => {
    const detail = formatLocalModelAssessmentFailure({
      _tag: "GeneratedClientTransportError",
      operationId: "assessModels",
      cause: new Error("connection failed"),
    })

    expect(detail).toContain("GeneratedClientTransportError")
    expect(detail).toContain("Error: connection failed")
    expect(detail).toContain("local-model-assessments.test.ts")
  })
})
