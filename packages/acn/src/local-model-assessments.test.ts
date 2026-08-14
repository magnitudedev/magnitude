import { Effect, Schema } from "effect"
import { describe, expect, it } from "vitest"

import {
  correlateLocalModelAssessmentProfiles,
  formatLocalModelAssessmentFailure,
  localModelAssessmentProfiles,
  localModelAssessmentResultFromIcn,
  performanceSampleContextTokens,
  type LocalModelAssessment,
} from "./local-model-assessments"
import {
  AssessmentEnvironmentIdSchema,
  ModelPackageSchema,
  ModelServingConfigurationIdSchema,
  ServableModelBundleSchema,
  type ServableModelBundle,
} from "@magnitudedev/acn-protocol"
import {
  ModelServingConfiguration as NativeModelServingConfigurationSchema,
  type ModelServingConfiguration as NativeModelServingConfiguration,
} from "@magnitudedev/icn-protocol/schemas"

const modelPackage = (id: string, maximumContextLength: number) =>
  Schema.decodeUnknownSync(ModelPackageSchema)({
    id,
    source: { _tag: "Local", path: `/models/${id}.gguf` },
    files: [{
      id: `file-${id}`,
      path: `${id}.gguf`,
      role: "weights",
      sizeBytes: 1,
      sha256: "a".repeat(64),
    }],
    relationships: [],
    properties: {
      format: "gguf",
      quantization: "Q4_K_M",
      quantizationName: "4-bit",
      architecture: "test",
      maximumContextLength,
    },
  })

const standaloneBundle = (maximumContextLength: number): ServableModelBundle =>
  Schema.validateSync(ServableModelBundleSchema)({
    _tag: "Standalone",
    package: modelPackage("package-test", maximumContextLength),
  })

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
    const target = Schema.validateSync(ServableModelBundleSchema)({
      _tag: "SpeculativeDecoding",
      target: modelPackage("target", 131_072),
      draftSource: {
        _tag: "Separate",
        draft: modelPackage("draft", 32_768),
      },
      method: { _tag: "DFlash" },
    })
    expect(localModelAssessmentProfiles(target)).toEqual([
      { contextLength: 32_768 },
    ])
  })

  it("caps separately paired speculative packages when both limits exceed 100K", () => {
    const target = Schema.validateSync(ServableModelBundleSchema)({
      _tag: "SpeculativeDecoding",
      target: modelPackage("target", 262_144),
      draftSource: {
        _tag: "Separate",
        draft: modelPackage("draft", 131_072),
      },
      method: { _tag: "DSpark" },
    })
    expect(localModelAssessmentProfiles(target)).toEqual([
      { contextLength: 100_000 },
    ])
  })

  it("uses only the target package limit for embedded speculative decoding", () => {
    const target = Schema.validateSync(ServableModelBundleSchema)({
      _tag: "SpeculativeDecoding",
      target: modelPackage("target", 80_000),
      draftSource: { _tag: "Embedded" },
      method: { _tag: "Mtp" },
    })
    expect(localModelAssessmentProfiles(target)).toEqual([
      { contextLength: 80_000 },
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
  const assessmentBundle = standaloneBundle(100_000)
  const nativeConfiguration = (
    id: string,
    contextLength: number,
  ): NativeModelServingConfiguration => Schema.validateSync(
    NativeModelServingConfigurationSchema,
  )({ id, bundle: assessmentBundle, profile: { contextLength } })

  it("preserves a request-local operational failure", () => {
    const failure = {
      code: "planning_worker_defect",
      message: "failed to create llama context",
      retryable: true,
    }
    const result = Effect.runSync(localModelAssessmentResultFromIcn({
      _tag: "Failed",
      requestId: "assessment-failed",
      failure,
    }, environmentId))

    expect(result).toEqual({ _tag: "Failed", failure })
  })

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

  it("correlates plural results into requested profile order", () => {
    const assessment = (contextLength: number): LocalModelAssessment => ({
      _tag: "Incompatible",
      configuration: {
        id: ModelServingConfigurationIdSchema.make(`configuration-${contextLength}`),
        bundle: assessmentBundle,
        profile: { contextLength },
      },
      failure: { code: "unsupported", message: "unsupported", retryable: false },
    })
    const result = Effect.runSync(correlateLocalModelAssessmentProfiles(
      [{ contextLength: 50_000 }, { contextLength: 100_000 }],
      [assessment(100_000), assessment(50_000)],
    ))

    expect(result.map(({ configuration }) => configuration.profile.contextLength))
      .toEqual([50_000, 100_000])
  })

  it("rejects malformed profile result sets as operation failures", () => {
    const assessment = (contextLength: number): LocalModelAssessment => ({
      _tag: "Incompatible",
      configuration: {
        id: ModelServingConfigurationIdSchema.make(`configuration-${contextLength}`),
        bundle: assessmentBundle,
        profile: { contextLength },
      },
      failure: { code: "unsupported", message: "unsupported", retryable: false },
    })
    const requested = [{ contextLength: 50_000 }, { contextLength: 100_000 }]

    for (const assessments of [
      [assessment(50_000)],
      [assessment(50_000), assessment(50_000), assessment(100_000)],
      [assessment(50_000), assessment(100_000), assessment(75_000)],
    ]) {
      expect(() => Effect.runSync(
        correlateLocalModelAssessmentProfiles(requested, assessments),
      )).toThrow()
    }
  })

  it("rejects duplicate requested profiles before assessment", () => {
    expect(() => Effect.runSync(correlateLocalModelAssessmentProfiles(
      [{ contextLength: 50_000 }, { contextLength: 50_000 }],
      [],
    ))).toThrow()
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
