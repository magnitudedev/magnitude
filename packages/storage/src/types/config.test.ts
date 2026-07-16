import { describe, expect, test } from "vitest"
import { Schema } from "effect"
import { MagnitudeConfigSchema } from "./config"

describe("MagnitudeConfig local inference and onboarding state", () => {
  test("keeps empty first-run config valid", () => {
    expect(Schema.decodeUnknownSync(MagnitudeConfigSchema)({})).toEqual({})
  })

  test("decodes independent versioned onboarding and desired local binding", () => {
    const value = {
      onboarding: {
        completions: {
          model_setup: {
            version: 2,
            completedAt: "2026-07-14T22:00:00.000Z",
          },
        },
      },
      localInference: {
        usage: {
          sessionConcurrency: "up_to_three",
        },
        binding: {
          _tag: "Managed",
          selectionId: "selection",
          artifactId: "artifact",
          providerModelId: "provider-model",
          contextTokens: 100_000,
          parallelSlots: 3,
        },
      },
    }
    expect(Schema.decodeUnknownSync(MagnitudeConfigSchema)(value)).toEqual(value)
  })

  test("rejects unversioned onboarding completion", () => {
    expect(() => Schema.decodeUnknownSync(MagnitudeConfigSchema)({
      onboarding: { completions: { model_setup: { completedAt: "now" } } },
    })).toThrow()
  })
})
