import { renderToStaticMarkup } from "react-dom/server"
import { Option, Schema } from "effect"
import { ChatModels } from "@magnitudedev/sdk"
import { describe, expect, it } from "vitest"
import { ChatThinkingSelect } from "./thinking"

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
const render = (m: ReturnType<typeof model> | undefined, effort?: string) =>
  renderToStaticMarkup(
    <ChatThinkingSelect
      model={m}
      selected={Option.fromNullable(effort)}
      onChange={() => {}}
    />
  )
const trigger = (html: string) =>
  html.match(/<button[^>]*aria-label="Thinking level"[^>]*>/)?.[0]

describe("Chat thinking control", () => {
  it("keeps an Off control visible and disabled without thinking support", () => {
    const html = render(model(null), "high")
    expect(trigger(html)).toMatch(/\sdisabled(?:=|\s|>)/)
    expect(html).toContain(">Off</span>")
    expect(html).toContain('tabindex="0"')
    expect(html).toContain('aria-label="Thinking level: Off"')
  })
  it("keeps the control visible before model selection", () => {
    const html = render(undefined)
    expect(trigger(html)).toMatch(/\sdisabled(?:=|\s|>)/)
    expect(html).toContain(">Off</span>")
  })
  it("shows the actual fixed mandatory level instead of incorrectly saying Off", () => {
    const html = render(
      model({
        ...profile,
        mandatory: true,
        default_effort: "high",
        supported_efforts: ["none", "high"],
      }),
      "none"
    )
    expect(trigger(html)).toMatch(/\sdisabled(?:=|\s|>)/)
    expect(html).toContain(">High</span>")
    expect(html).not.toContain(">Off</span>")
  })
  it("enables the dropdown for configurable levels and preserves selection", () => {
    for (const level of ["none", "low", "medium", "high"]) {
      const html = render(model(profile), level)
      expect(trigger(html)).toBeDefined()
      expect(trigger(html)).not.toMatch(/\sdisabled(?:=|\s|>)/)
      expect(html).not.toContain('aria-label="Thinking level:')
      const label =
        level === "none" ? "Off" : level[0]!.toUpperCase() + level.slice(1)
      expect(html).toContain(`>${label}</span>`)
    }
  })
  it("uses the new model default when a previous level is unsupported", () => {
    const html = render(model(profile), "maximum")
    expect(html).toContain(">Medium</span>")
    expect(trigger(html)).not.toMatch(/\sdisabled(?:=|\s|>)/)
  })
  it("keeps an explicitly Off-only capability disabled", () => {
    const html = render(
      model({ ...profile, default_effort: "none", supported_efforts: ["none"] })
    )
    expect(trigger(html)).toMatch(/\sdisabled(?:=|\s|>)/)
    expect(html).toContain(">Off</span>")
  })
})
