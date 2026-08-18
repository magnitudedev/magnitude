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
        modelOptionsState={{
          _tag: "Ready",
          options: [{ value: selectedModelId, label: "Qwen Test (Q4)" }],
        }}
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
        modelOptionsState={{
          _tag: "Ready",
          options: [{ value: selectedModelId, label: "Qwen Test" }],
        }}
        selectedModelId={selectedModelId}
        onModelSelect={() => {}}
      />
    )

    expect(html).toContain("enabled:hover:bg-slate-100")
    expect(html).toContain("dark:enabled:hover:bg-slate-750")
    expect(html).not.toContain("border-slate-300")
    expect(html).not.toContain("dark:bg-slate-900")
    expect(html).not.toContain("shadow-")
    expect(html).not.toContain("hover:text-")
    expect(html).not.toContain("hover:underline")
  })

  it("keeps a selected model at full foreground while choices refresh", () => {
    const html = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Qwen Test"
        modelOptionsState={{ _tag: "Loading", options: [] }}
        selectedModelId={selectedModelId}
        onModelSelect={() => {}}
      />
    )

    expect(html).toContain("text-slate-900")
    expect(html).toContain("dark:text-slate-50")
    expect(html).toContain("data-placeholder:text-slate-900")
    expect(html).toContain("dark:data-placeholder:text-slate-50")
    expect(html).not.toContain(' disabled=""')
    expect(html).not.toContain(' data-disabled=""')
    expect(html).not.toContain(' data-placeholder=""')
    expect(html).toContain('aria-busy="true"')
  })

  it("shows authoritative model loading before an initial selection is available", () => {
    const html = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Choose model"
        modelOptionsState={{ _tag: "Loading", options: [] }}
        selectedModelId={null}
        onModelSelect={() => {}}
      />
    )

    expect(html).toContain('aria-label="Model: Loading models…"')
    expect(html).toContain('aria-busy="true"')
    expect(html).toContain("animate-spin")
    expect(html).not.toContain(' disabled=""')
  })

  it("distinguishes model loading failure from a loaded empty model list", () => {
    const failed = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Choose model"
        modelOptionsState={{ _tag: "Failed", options: [] }}
        selectedModelId={null}
        onModelSelect={() => {}}
      />
    )
    const empty = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Choose model"
        modelOptionsState={{ _tag: "Ready", options: [] }}
        selectedModelId={null}
        onModelSelect={() => {}}
      />
    )

    expect(failed).toContain('aria-label="Model: Models unavailable"')
    expect(failed).not.toContain('aria-busy="true"')
    expect(empty).toContain('aria-label="Model: Choose model"')
    expect(empty).toContain(' disabled=""')
  })
})
