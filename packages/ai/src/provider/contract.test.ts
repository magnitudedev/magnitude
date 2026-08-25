import { describe, expectTypeOf, it } from "vitest"
import type { BaseCallOptions } from "./call-options"
import type { BoundModel } from "../model/bound-model"
import type { StreamStartFailure } from "../errors/failure"
import type { ModelPreparationObserver } from "../model/model-spec"

describe("model stream activity types", () => {
  it("omits Preparing by default and preserves a consumer-defined payload", () => {
    type Preparation = { readonly kind: "loading"; readonly fraction: number }
    type PlainStream = BoundModel<BaseCallOptions>["stream"]
    type PreparingStream = BoundModel<
      BaseCallOptions,
      StreamStartFailure,
      Preparation
    >["stream"]
    expectTypeOf<Parameters<PlainStream>[3]>().toEqualTypeOf<
      ModelPreparationObserver<never> | undefined
    >()
    expectTypeOf<Parameters<PreparingStream>[3]>().toEqualTypeOf<
      ModelPreparationObserver<Preparation> | undefined
    >()
    expectTypeOf<Parameters<NonNullable<Parameters<PreparingStream>[3]>>[0]>()
      .toEqualTypeOf<{
      readonly preparation: Preparation
      readonly requestId: string | null
    }>()
  })
})
