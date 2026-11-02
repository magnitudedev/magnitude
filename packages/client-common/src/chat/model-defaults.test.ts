import { Option, Schema } from "effect"
import { InferenceModelsResponseSchema, ReasoningEffortSchema, ServingModelId } from "@magnitudedev/sdk"
import { describe, expect, it } from "vitest"
import { defaultResponseSettings } from "./draft"
import {
  changedChatDefaults,
  chatDefaultsStatus,
  chatModelDefaults,
  chatStartingSettings,
} from "./model-defaults"

const model = Schema.decodeUnknownSync(InferenceModelsResponseSchema)({
  object: "list",
  data: [
    {
      id: "qwen3.5-4b:gguf:q4",
      name: "Qwen3.5 4B",
      description: "",
      created: 0,
      object: "model",
      owned_by: "local",
      context_length: 32768,
      architecture: { input_modalities: ["text"], output_modalities: ["text"] },
      top_provider: { context_length: 32768, max_completion_tokens: 32768 },
      supported_parameters: [],
      reasoning: {
        mandatory: false,
        default_enabled: true,
        default_effort: "high",
        supported_efforts: ["none", "high"],
      },
    },
  ],
}).data[0]!
const id = ServingModelId.make(model.id)
const off = ReasoningEffortSchema.make("none")

describe("Chat model defaults", () => {
  it("starts a model with nothing saved from the standard settings and its default thinking", () => {
    const start = chatStartingSettings(model, {})
    expect(start).toMatchObject({ ...defaultResponseSettings, model: Option.some(id) })
    expect(start.reasoningEffort).toEqual(Option.some("high"))
    expect(chatDefaultsStatus(model, {}, changedChatDefaults(start, model, {}))).toBe("standard")
  })

  it("saves every setting and the thinking level, so thinking off survives", () => {
    const settings = {
      ...chatStartingSettings(model, {}),
      systemPrompt: "Answer briefly.",
      temperature: Option.some(0.7),
      seed: Option.some(42),
      stop: ["</answer>"],
      reasoningEffort: Option.some(off),
    }
    const saved = { [id]: chatModelDefaults(settings, model) }
    expect(saved[id].reasoningEffort).toEqual(Option.some("none"))
    expect(chatStartingSettings(model, saved)).toEqual(settings)
    expect(chatDefaultsStatus(model, saved, changedChatDefaults(settings, model, saved))).toBe("yours")
  })

  it("names each setting that differs from where the chat started", () => {
    const start = chatStartingSettings(model, {})
    const changed = changedChatDefaults(
      { ...start, temperature: Option.some(0.7), stop: ["END"], reasoningEffort: Option.some(off) },
      model,
      {}
    )
    expect([...changed].sort()).toEqual(["reasoningEffort", "stop", "temperature"])
    expect(chatDefaultsStatus(model, {}, changed)).toBe("changedFromStandard")
  })

  it("compares against the saved defaults once a model has them", () => {
    const saved = { [id]: chatModelDefaults({ ...chatStartingSettings(model, {}), temperature: Option.some(0.7) }, model) }
    const changed = changedChatDefaults({ ...chatStartingSettings(model, saved), temperature: Option.some(1.2) }, model, saved)
    expect([...changed]).toEqual(["temperature"])
    expect(chatDefaultsStatus(model, saved, changed)).toBe("changedFromYours")
  })

  it("treats an unset thinking level as the level the model would use", () => {
    const start = chatStartingSettings(model, {})
    expect(changedChatDefaults({ ...start, reasoningEffort: Option.none() }, model, {}).size).toBe(0)
  })
})
