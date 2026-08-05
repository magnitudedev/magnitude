import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"

import {
  clearAssessmentLifecycle,
  completeAssessmentLifecycle,
  formatLocalModelAssessmentFailure,
  localModelAssessmentResultFromIcn,
  modelAssessmentProfiles,
} from "./local-model-assessments"
import {
  AssessmentEnvironmentIdSchema,
  ModelAssessmentIdSchema,
  ModelOfferingTargetIdSchema,
  ModelServingConfigurationIdSchema,
  type ModelOfferingTarget,
} from "@magnitudedev/acn-protocol"

const packageTarget = (maximumContextLength: number): ModelOfferingTarget => ({
  _tag: "Package",
  package: { properties: { maximumContextLength } },
} as unknown as ModelOfferingTarget)

describe("modelAssessmentProfiles", () => {
  it("includes standard points through the model limit plus the exact limit", () => {
    expect(modelAssessmentProfiles(packageTarget(131_072))).toEqual([
      { contextLength: 100_000 },
      { contextLength: 131_072 },
    ])
  })

  it("bounds, deduplicates, and preserves the exact model maximum", () => {
    expect(modelAssessmentProfiles(packageTarget(80_000))).toEqual([
      { contextLength: 80_000 },
    ])
    expect(modelAssessmentProfiles(packageTarget(200_000))).toEqual([
      { contextLength: 100_000 },
      { contextLength: 200_000 },
    ])
    expect(modelAssessmentProfiles(packageTarget(300_000))).toEqual([
      { contextLength: 100_000 },
      { contextLength: 200_000 },
      { contextLength: 300_000 },
    ])
  })

  it("uses the lower package limit for speculative decoding", () => {
    const target = {
      _tag: "SpeculativeDecodingPair",
      target: { properties: { maximumContextLength: 131_072 } },
      draft: { properties: { maximumContextLength: 32_768 } },
    } as unknown as ModelOfferingTarget
    expect(modelAssessmentProfiles(target)).toEqual([
      { contextLength: 32_768 },
    ])
  })

  it("does not invent a profile below the product minimum", () => {
    expect(modelAssessmentProfiles(packageTarget(2_048))).toEqual([])
  })
})

describe("assessment lifecycle", () => {
  const targetId = ModelOfferingTargetIdSchema.make("target-test")
  const current = new Map([[targetId, {
    _tag: "Assessing" as const,
  }]])

  it("clears active state when its serialized owner exits", () => {
    expect(clearAssessmentLifecycle(current, [targetId]).get(targetId)).toEqual({
      _tag: "Unassessed",
    })
  })

  it("terminalizes completed assessment state", () => {
    const configurationId = ModelServingConfigurationIdSchema.make("configuration-test")
    const completed = [{
      _tag: "Assessed" as const,
      targetId,
      environmentId: AssessmentEnvironmentIdSchema.make("environment-test"),
      assessments: [{
        _tag: "DoesNotFit" as const,
        profile: { contextLength: 100_000 },
        configurationId,
        assessmentId: ModelAssessmentIdSchema.make("assessment-test"),
        memory: [],
        deficitBytes: 1,
        limitingResource: "system",
      }],
    }]
    expect(completeAssessmentLifecycle(
      current,
      [targetId],
      completed,
    ).get(targetId)).toEqual({
      _tag: "Assessed",
      environmentId: "environment-test",
      configurationIds: [configurationId],
    })
  })
})

describe("localModelAssessmentResultFromIcn", () => {
  const environmentId = AssessmentEnvironmentIdSchema.make("environment-test")

  it("preserves terminal non-capacity assessment evidence", () => {
    const result = Effect.runSync(localModelAssessmentResultFromIcn({
      _tag: "Assessed",
      requestId: "assessment-0",
      targetId: "target-0",
      profiles: [{
        _tag: "DoesNotFit",
        profile: { contextLength: 50_000 },
        configurationId: "configuration-0",
        assessmentId: "assessment-result-0",
        memory: [],
        limitingResource: "system_memory",
        deficitBytes: 1024,
      }, {
        _tag: "Incompatible",
        profile: { contextLength: 100_000 },
        configurationId: "configuration-1",
        failure: {
          code: "unsupported_architecture",
          message: "Unsupported architecture",
          retryable: false,
        },
      }],
    }, environmentId))

    expect(result).toEqual({
      _tag: "Assessed",
      targetId: "target-0",
      environmentId,
      assessments: [{
        _tag: "DoesNotFit",
        profile: { contextLength: 50_000 },
        configurationId: "configuration-0",
        assessmentId: "assessment-result-0",
        memory: [],
        limitingResource: "system_memory",
        deficitBytes: 1024,
      }, {
        _tag: "Incompatible",
        profile: { contextLength: 100_000 },
        configurationId: "configuration-1",
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
