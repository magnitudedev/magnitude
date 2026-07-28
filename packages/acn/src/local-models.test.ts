import { describe, expect, it } from "vitest"
import { Either, Option } from "effect"
import {
  ModelOfferingTargetIdSchema,
  ProviderModelIdSchema,
  type LocalModelsState,
} from "@magnitudedev/sdk"
import {
  localModelTargetPreparationOutcome,
  preparationFromProviderProjection,
  preparationFromReconciliationFailure,
} from "./local-models"

describe("local model preparation", () => {
  const providerModelIds = [ProviderModelIdSchema.make("test-configuration")]

  it("keeps retryable automatic reconciliation failures in preparation", () => {
    expect(preparationFromReconciliationFailure(providerModelIds, {
      code: "local_offering_assessment_unavailable",
      message: "This model is not yet configured",
      retryable: true,
    })).toEqual({ _tag: "Preparing" })
  })

  it("exposes terminal reconciliation failures as unavailable", () => {
    const failure = {
      code: "invalid_target",
      message: "The model files are incompatible",
      retryable: false,
    }
    expect(preparationFromReconciliationFailure(providerModelIds, failure)).toEqual({
      _tag: "Unavailable",
      providerModelIds,
      failure,
    })
  })

  it("keeps provider availability in preparation until it matches the package snapshot", () => {
    expect(preparationFromProviderProjection(
      providerModelIds,
      new Map([[providerModelIds[0]!, {
        availability: { _tag: "Disabled", reason: "incompatible_runtime" },
      }]]),
      false,
      Option.none(),
    )).toEqual({ _tag: "Preparing" })
  })

  it("exposes an authoritative current provider incompatibility", () => {
    expect(preparationFromProviderProjection(
      providerModelIds,
      new Map([[providerModelIds[0]!, {
        availability: { _tag: "Disabled", reason: "incompatible_runtime" },
      }]]),
      true,
      Option.none(),
    )).toEqual({
      _tag: "Unavailable",
      providerModelIds,
      failure: {
        code: "incompatible_runtime",
        message: "This model configuration is not available to the local runtime",
        retryable: true,
      },
    })
  })
})

describe("local model target preparation outcome", () => {
  const targetId = ModelOfferingTargetIdSchema.make("target")
  const state = (preparation: LocalModelsState["models"][number]["preparation"]): LocalModelsState => ({
    models: [{
      targetId,
      catalogCandidateIds: [],
      providerModelIds: [],
      displayName: "Test",
      description: "",
      kind: "Standalone",
      quantization: "Q4",
      maximumContextLength: 1,
      downloadBytes: 1,
      download: { _tag: "Downloaded", installedBytes: 1 },
      preparation,
    }],
    recommendations: { _tag: "Ready", entries: [], catalog: [], progress: [] },
  })

  it("waits while the installed target is still preparing", () => {
    expect(Option.isNone(localModelTargetPreparationOutcome(
      state({ _tag: "Preparing" }),
      targetId,
    ))).toBe(true)
  })

  it("completes only when the target is available to load", () => {
    const outcome = localModelTargetPreparationOutcome(
      state({ _tag: "Available", providerModelIds: [] }),
      targetId,
    )
    expect(Option.isSome(outcome) && Either.isRight(outcome.value)).toBe(true)
  })

  it("preserves terminal preparation failure", () => {
    const outcome = localModelTargetPreparationOutcome(
      state({
        _tag: "Unavailable",
        providerModelIds: [],
        failure: {
          code: "invalid_target",
          message: "The downloaded model is invalid",
          retryable: false,
        },
      }),
      targetId,
    )
    expect(Option.isSome(outcome) && Either.isLeft(outcome.value)
      ? outcome.value.left
      : null).toMatchObject({
        code: "invalid_target",
        message: "The downloaded model is invalid",
        retryable: false,
      })
  })
})
