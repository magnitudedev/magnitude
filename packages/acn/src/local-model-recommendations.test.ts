import { LocalModelMutationFailed } from "@magnitudedev/acn-protocol"
import { describe, expect, it } from "vitest"

import { localModelRecommendationFailure } from "./local-model-recommendations"

describe("localModelRecommendationFailure", () => {
  it("preserves typed assessment failure metadata for the public lifecycle", () => {
    expect(localModelRecommendationFailure(new LocalModelMutationFailed({
      code: "planner_timeout",
      message: "Hardware assessment took longer than five minutes.",
      retryable: true,
    }))).toEqual({
      code: "planner_timeout",
      message: "Hardware assessment took longer than five minutes.",
      retryable: true,
    })
  })
})
