import { describe, expect, test } from "vitest"
import { Schema } from "effect"
import { MagnitudeConfigSchema } from "./config"

describe("MagnitudeConfig onboarding state", () => {
  test("keeps old config files valid", () => {
    expect(Schema.decodeUnknownSync(MagnitudeConfigSchema)({})).toEqual({})
  })

  test("decodes the versioned CLI model-setup marker", () => {
    expect(Schema.decodeUnknownSync(MagnitudeConfigSchema)({
      onboarding: {
        cliModelSetupVersion: 2,
        completedAt: "2026-07-14T22:00:00.000Z",
      },
    })).toEqual({
      onboarding: {
        cliModelSetupVersion: 2,
        completedAt: "2026-07-14T22:00:00.000Z",
      },
    })
  })
})
