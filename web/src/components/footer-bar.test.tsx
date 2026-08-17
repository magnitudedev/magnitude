import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/sdk"
import { FooterBar } from "./footer-bar"

const selectedModelId = ProviderModelIdSchema.make("qwen-test-q4")
const high = ReasoningEffortSchema.make("high")

describe("FooterBar", () => {
  it("keeps runtime fields separate and in footer order", () => {
    const html = renderToStaticMarkup(
      <FooterBar
        context={{ tokenEstimate: 5_000, isCompacting: false }}
        tokenCap={100_000}
        model="Qwen Test (Q4)"
        thinkingLevel="High"
        memoryLabel="16 GB mem"
        modelOptions={[{ value: selectedModelId, label: "Qwen Test (Q4)" }]}
        selectedModelId={selectedModelId}
        onModelSelect={() => {}}
        thinkingOptions={[{ value: high, label: "High" }]}
        thinkingEffort={high}
        onThinkingSelect={() => {}}
        onMemoryClick={() => {}}
      />
    )

    expect(html).toContain('aria-label="Model: Qwen Test (Q4)"')
    expect(html).toContain('aria-label="Thinking level: High"')
    expect(html).toContain("Qwen Test (Q4)")
    expect(html.match(/aria-haspopup="listbox"/g)).toHaveLength(2)
    expect(html.match(/aria-expanded="false"/g)).toHaveLength(2)
    expect(html).toContain("<svg")
    expect(html).not.toContain("hover:underline")
    expect(html).toContain(">16 GB mem</button>")

    const model = html.indexOf("Qwen Test (Q4)")
    const thinking = html.indexOf("High")
    const memory = html.indexOf("16 GB mem")
    const context = html.indexOf("95% remaining")
    expect(model).toBeGreaterThan(-1)
    expect(model).toBeLessThan(thinking)
    expect(thinking).toBeLessThan(memory)
    expect(memory).toBeLessThan(context)
    expect(html).not.toContain("/Users/test/magnitude")
  })

  it("uses background-only hover treatment for the model controls", () => {
    const html = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Qwen Test"
        modelOptions={[{ value: selectedModelId, label: "Qwen Test" }]}
        selectedModelId={selectedModelId}
        onModelSelect={() => {}}
      />
    )

    expect(html).toContain("enabled:hover:bg-slate-100")
    expect(html).toContain("dark:enabled:hover:bg-slate-750")
    expect(html).not.toContain("hover:text-")
    expect(html).not.toContain("hover:underline")
  })
})
