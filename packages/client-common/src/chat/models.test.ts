import { Option, Schema } from "effect"
import { ChatModels } from "@magnitudedev/sdk"
import { describe, expect, it } from "vitest"
import { chatReasoningEffort, chatReasoningLevels } from "./models"
const model = (reasoning: unknown) =>
  Schema.decodeUnknownSync(ChatModels)({
    object: "list",
    data: [
      {
        id: "test",
        name: "Test",
        description: "",
        created: 0,
        object: "model",
        owned_by: "local",
        context_length: 32768,
        architecture: {
          input_modalities: ["text"],
          output_modalities: ["text"],
        },
        top_provider: { context_length: 32768, max_completion_tokens: 32768 },
        supported_parameters: [],
        reasoning,
      },
    ],
  }).data[0]!
const profile = {
  mandatory: false,
  default_enabled: true,
  default_effort: "medium",
  supported_efforts: ["none", "low", "medium", "high"],
}
describe("chat reasoning capabilities", () => {
  it("shows supported levels with Off mapped to the engine's none wire value", () => {
    expect(chatReasoningLevels(model(profile))).toEqual([
      { value: "none", label: "Off" },
      { value: "low", label: "Low" },
      { value: "medium", label: "Medium" },
      { value: "high", label: "High" },
    ])
  })
  it("never exposes Off for mandatory reasoning", () => {
    expect(
      chatReasoningLevels(model({ ...profile, mandatory: true })).map(
        (level) => level.value
      )
    ).toEqual(["low", "medium", "high"])
  })
  it("uses the declared default and retains valid selections across models", () => {
    const m = model(profile)
    expect(chatReasoningEffort(m, Option.none())).toEqual(Option.some("medium"))
    expect(chatReasoningEffort(m, Option.some("high"))).toEqual(
      Option.some("high")
    )
    expect(chatReasoningEffort(m, Option.some("maximum"))).toEqual(
      Option.some("medium")
    )
  })
  it("omits the request value for models without reasoning", () => {
    expect(chatReasoningLevels(model(null))).toEqual([])
    expect(chatReasoningEffort(model(null), Option.some("high"))).toEqual(
      Option.none()
    )
    expect(chatReasoningLevels(undefined)).toEqual([])
  })
})
