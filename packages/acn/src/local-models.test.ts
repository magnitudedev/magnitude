import { describe, expect, it } from "vitest"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { preparationFromReconciliationFailure } from "./local-models"

describe("local model preparation", () => {
  const providerModelIds = [ProviderModelIdSchema.make("local:test")]

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
})
