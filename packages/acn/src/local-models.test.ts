import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ProviderModelIdSchema,
} from "@magnitudedev/sdk"
import {
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
