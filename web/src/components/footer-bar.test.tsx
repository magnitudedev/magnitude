import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/sdk"
import { FooterBar } from "./footer-bar"

const selectedModelId = ProviderModelIdSchema.make("qwen-test-q4")
const high = ReasoningEffortSchema.make("high")
const modelOption = {
  value: selectedModelId,
  label: "Qwen Test (Q4)",
  thinkingOptions: [{ value: high, label: "High" }],
  defaultThinkingEffort: high,
} as const

describe("FooterBar", () => {
  it("places context immediately before the combined model and thinking control", () => {
    const html = renderToStaticMarkup(
      <FooterBar
        context={{ tokenEstimate: 5_000, isCompacting: false }}
        showContext
        tokenCap={100_000}
        model="Qwen Test (Q4)"
        thinkingLevel="High"
        memoryLabel="16 GB mem"
        modelOptionsState={{
          _tag: "Ready",
          options: [modelOption],
        }}
        selectedModelId={selectedModelId}
        onSelectionCommit={() => {}}
        thinkingOptions={[{ value: high, label: "High" }]}
        thinkingEffort={high}
        onThinkingSelect={() => {}}
        onMemoryClick={() => {}}
      />
    )

    expect(html).toContain('aria-label="Model: Qwen Test (Q4). Thinking level: High"')
    expect(html).toContain("Qwen Test (Q4)")
    expect(html.match(/aria-haspopup="menu"/g)).toHaveLength(1)
    expect(html).toContain("<svg")
    expect(html).not.toContain("hover:underline")
    expect(html).toContain(">16 GB mem</button>")

    const memory = html.indexOf("16 GB mem")
    const context = html.indexOf("95% remaining")
    const model = html.indexOf("Qwen Test (Q4)")
    const thinking = html.indexOf("High")
    expect(memory).toBeGreaterThan(-1)
    expect(memory).toBeLessThan(context)
    expect(context).toBeLessThan(model)
    expect(model).toBeGreaterThan(-1)
    expect(model).toBeLessThan(thinking)
    expect(html).not.toContain("/Users/test/magnitude")
  })

  it("hides context until the chat has at least one message", () => {
    const empty = renderToStaticMarkup(
      <FooterBar context={null} model="Qwen Test" />
    )
    const started = renderToStaticMarkup(
      <FooterBar context={null} showContext model="Qwen Test" />
    )

    expect(empty).not.toContain("100% remaining")
    expect(started).toContain("100% remaining")
  })

  it("uses background-only hover treatment for the model controls", () => {
    const html = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Qwen Test"
        modelOptionsState={{
          _tag: "Ready",
          options: [{ ...modelOption, label: "Qwen Test" }],
        }}
        selectedModelId={selectedModelId}
        onSelectionCommit={() => {}}
      />
    )

    expect(html).toContain("hover:bg-slate-150")
    expect(html).toContain("dark:hover:bg-slate-750")
    expect(html).not.toContain("border-slate-300")
    expect(html).not.toContain("dark:bg-slate-900")
    expect(html).not.toContain("hover:text-")
    expect(html).not.toContain("hover:underline")
  })

  it("hides thinking for unsupported models and names a supported none effort", () => {
    const unsupported = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Embedding Model"
        modelOptionsState={{
          _tag: "Ready",
          options: [{ ...modelOption, label: "Embedding Model", thinkingOptions: [] }],
        }}
        selectedModelId={selectedModelId}
        thinkingLevel={null}
      />
    )
    const none = ReasoningEffortSchema.make("none")
    const supported = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Qwen Test"
        modelOptionsState={{
          _tag: "Ready",
          options: [{
            ...modelOption,
            label: "Qwen Test",
            thinkingOptions: [{ value: none, label: "None" }],
          }],
        }}
        selectedModelId={selectedModelId}
        thinkingLevel="None"
        thinkingOptions={[{ value: none, label: "None" }]}
        thinkingEffort={none}
      />
    )

    expect(unsupported).toContain('aria-label="Model: Embedding Model"')
    expect(unsupported).not.toContain("Thinking level:")
    expect(supported).toContain('aria-label="Model: Qwen Test. Thinking level: None"')
  })

  it("keeps a selected model at full foreground while choices refresh", () => {
    const html = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Qwen Test"
        modelOptionsState={{ _tag: "Loading", options: [] }}
        selectedModelId={selectedModelId}
        onSelectionCommit={() => {}}
      />
    )

    expect(html).toContain("text-slate-900")
    expect(html).toContain("dark:text-slate-50")
    expect(html).not.toContain(' disabled=""')
    expect(html).not.toContain(' data-disabled=""')
    expect(html).not.toContain(' data-placeholder=""')
    expect(html).toContain('aria-busy="true"')
  })

  it("keeps a selected model at full foreground when the menu has no actions", () => {
    const html = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Qwen Test"
        modelOptionsState={{ _tag: "Ready", options: [] }}
        selectedModelId={selectedModelId}
        onSelectionCommit={() => {}}
      />
    )

    expect(html).toContain('aria-label="Model: Qwen Test"')
    expect(html).not.toContain(' disabled=""')
    expect(html).not.toContain("disabled:opacity-100")
    expect(html).not.toContain('aria-haspopup="menu"')
  })

  it("shows authoritative model loading before an initial selection is available", () => {
    const html = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Choose model"
        modelOptionsState={{ _tag: "Loading", options: [] }}
        selectedModelId={null}
        onSelectionCommit={() => {}}
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
        onSelectionCommit={() => {}}
      />
    )
    const empty = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Choose model"
        modelOptionsState={{ _tag: "Ready", options: [] }}
        selectedModelId={null}
        onSelectionCommit={() => {}}
      />
    )

    expect(failed).toContain('aria-label="Model: Models unavailable"')
    expect(failed).not.toContain('aria-busy="true"')
    expect(empty).toContain('aria-label="Model: Choose model"')
    expect(empty).toContain(' disabled=""')
  })
})
