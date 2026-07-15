import { describe, expect, test } from "vitest"
import { Schema } from "effect"
import { MagnitudeConfigSchema } from "./config"

describe("MagnitudeConfig onboarding state", () => {
  test("keeps old config files valid", () => {
    expect(Schema.decodeUnknownSync(MagnitudeConfigSchema)({})).toEqual({})
  })

  test("decodes the CLI model-setup completion marker and local usage", () => {
    expect(Schema.decodeUnknownSync(MagnitudeConfigSchema)({
      onboarding: {
        completedAt: "2026-07-14T22:00:00.000Z",
      },
      localInference: {
        localModelRole: "main",
        sessionConcurrency: "up_to_three",
      },
    })).toEqual({
      onboarding: {
        completedAt: "2026-07-14T22:00:00.000Z",
      },
      localInference: {
        localModelRole: "main",
        sessionConcurrency: "up_to_three",
      },
    })
  })
})
