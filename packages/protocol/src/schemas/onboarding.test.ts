import { describe, expect, test } from "vitest"
import { Schema } from "effect"
import { OnboardingState } from "./onboarding"

describe("onboarding protocol schema", () => {
  test("contains only durable completion", () => {
    const state = { completed: false }
    expect(Schema.decodeUnknownSync(OnboardingState)(state)).toEqual(state)
  })
})
