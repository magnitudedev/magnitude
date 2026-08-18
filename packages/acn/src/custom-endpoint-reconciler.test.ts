import { Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  CustomEndpointDeclarationsSchema,
  type CustomEndpointDeclarations,
} from "@magnitudedev/storage"
import { removedCustomEndpointModels } from "./custom-endpoint-reconciler"

const declarations = (
  models: readonly string[],
  displayName = "OpenRouter",
): CustomEndpointDeclarations =>
  Schema.decodeUnknownSync(CustomEndpointDeclarationsSchema)({
    openrouter: {
      displayName,
      connection: {
        baseUrl: "https://openrouter.ai/api/v1",
        authentication: { type: "none" },
      },
      models: Object.fromEntries(models.map((model) => [model, {
        displayName: model,
        contextWindow: 1048576,
        maxOutputTokens: 128000,
      }])),
    },
  })

describe("custom endpoint removal reconciliation", () => {
  it("identifies only provider/model identities removed from declarations", () => {
    expect(removedCustomEndpointModels(
      declarations(["z-ai/glm-5.2", "moonshotai/kimi-k2"]),
      declarations(["z-ai/glm-5.2"]),
    )).toEqual([{
      providerId: "custom:openrouter",
      providerModelId: "moonshotai/kimi-k2",
    }])
  })

  it("does not treat declaration metadata changes as model removal", () => {
    const previous = declarations(["z-ai/glm-5.2"])
    const next = declarations(["z-ai/glm-5.2"], "OpenRouter updated")

    expect(removedCustomEndpointModels(previous, next)).toEqual([])
  })

  it("identifies every model when an endpoint is removed", () => {
    expect(removedCustomEndpointModels(
      declarations(["z-ai/glm-5.2"]),
      {},
    )).toEqual([{
      providerId: "custom:openrouter",
      providerModelId: "z-ai/glm-5.2",
    }])
  })
})
