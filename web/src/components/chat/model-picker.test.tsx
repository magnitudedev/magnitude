import { renderToStaticMarkup } from "react-dom/server"
import { Option, Schema } from "effect"
import { InferenceModelsResponseSchema, ReasoningEffortSchema } from "@magnitudedev/sdk"
import { describe, expect, it } from "vitest"
import { ModelPicker, thinkingSuffix } from "./model-picker"

const model = (reasoning: unknown) =>
  Schema.decodeUnknownSync(InferenceModelsResponseSchema)({
    object: "list",
    data: [{
      id: "test", name: "Test", description: "", created: 0, object: "model", owned_by: "local", context_length: 32768,
      architecture: { input_modalities: ["text"], output_modalities: ["text"] },
      top_provider: { context_length: 32768, max_completion_tokens: 32768 },
      supported_parameters: ["tools"],
      reasoning,
    }],
  }).data[0]!
const profile = { mandatory: false, default_enabled: true, default_effort: "medium", supported_efforts: ["none", "low", "medium", "high"] }
const effort = (value?: string) => Option.map(Option.fromNullable(value), ReasoningEffortSchema.make)

describe("model and thinking picker", () => {
  it("names the thinking level after the model, says when thinking is off, and says nothing without thinking", () => {
    expect(thinkingSuffix(model(profile), effort("high"))).toBe("High")
    expect(thinkingSuffix(model(profile), effort())).toBe("Medium")
    expect(thinkingSuffix(model(profile), effort("none"))).toBe("Thinking off")
    expect(thinkingSuffix(model(null), effort("high"))).toBeNull()
    expect(thinkingSuffix(undefined, effort("high"))).toBeNull()
  })

  it("is one control: the model's name and its thinking level in a single trigger, without chevrons", () => {
    const m = model(profile)
    const html = renderToStaticMarkup(<ModelPicker models={[{ model: m, name: "Test (Q4)", context: 32768, note: "Loaded" }]} model={m} name="Test (Q4)"
      selected={effort("high")} onModel={() => {}} onEffort={() => {}} onManage={() => {}} />)
    expect(html.match(/<button/g)).toHaveLength(1)
    expect(html).toContain('aria-label="Model and thinking: Test (Q4), High"')
    expect(html).toContain(">High</span>")
    expect(html).not.toContain("·")
    expect(html).not.toContain("svg class=\"pointer-events-none size-4")
  })
})
