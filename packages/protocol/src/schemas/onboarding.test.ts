import { describe, expect, test } from "vitest"
import { Schema } from "effect"
import { OnboardingState } from "./onboarding"

describe("onboarding protocol schema", () => {
  test("contains only generic versioned flow completion", () => {
    const state = {
      flows: {
        model_setup: {
          currentVersion: 1,
          completedVersion: null,
          completedAt: null,
          required: true,
        },
      },
    }
    expect(Schema.decodeUnknownSync(OnboardingState)(state)).toEqual(state)
  })
})
