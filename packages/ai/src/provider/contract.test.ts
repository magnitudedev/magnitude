import { describe, expectTypeOf, it } from "vitest"
import type { ModelStreamEvent } from "../model/model-spec"

describe("model stream activity types", () => {
  it("omits Preparing by default and preserves a consumer-defined payload", () => {
    type Preparation = { readonly kind: "loading"; readonly fraction: number }
    expectTypeOf<Extract<ModelStreamEvent, { readonly _tag: "preparation_update" }>>()
      .toEqualTypeOf<never>()
    expectTypeOf<Extract<
      ModelStreamEvent<Preparation>,
      { readonly _tag: "preparation_update" }
    >>().toEqualTypeOf<{
      readonly _tag: "preparation_update"
      readonly preparation: Preparation
      readonly requestId: string | null
    }>()
  })
})
