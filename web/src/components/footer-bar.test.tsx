import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { FooterBar } from "./footer-bar"
import { formatFooterContextUsage } from "./local-inference-format"

describe("formatFooterContextUsage", () => {
  it("matches the CLI token and percentage presentation", () => {
    expect(
      formatFooterContextUsage(
        {
          tokenEstimate: 15_800,
          isCompacting: false,
        },
        200_000
      )
    ).toBe("16k / 200k (8%)")
  })

  it("shows the known capacity before the session has token usage", () => {
    expect(formatFooterContextUsage(null, 100_000)).toBe("— / 100k")
  })
})

describe("FooterBar", () => {
  it("keeps runtime fields separate and in footer order", () => {
    const html = renderToStaticMarkup(
      <FooterBar
        context={{ tokenEstimate: 5_000, isCompacting: false }}
        tokenCap={100_000}
        cwd="/Users/test/magnitude"
        model="Qwen Test (Q4)"
        modelResidency="ready"
        thinkingLevel="High"
        memoryLabel="16 GB mem"
        onModelClick={() => {}}
        onMemoryClick={() => {}}
      />
    )

    expect(html).toContain('aria-label="Model ready"')
    expect(html).toContain(">Qwen Test (Q4)</button>")
    expect(html).toContain('aria-label="Reasoning effort: High"')
    expect(html).toContain(">16 GB mem</button>")

    const model = html.indexOf("Qwen Test (Q4)")
    const thinking = html.indexOf("High")
    const memory = html.indexOf("16 GB mem")
    const context = html.indexOf("5k / 100k (5%)")
    const cwd = html.indexOf("/Users/test/magnitude")
    expect(model).toBeGreaterThan(-1)
    expect(model).toBeLessThan(thinking)
    expect(thinking).toBeLessThan(memory)
    expect(memory).toBeLessThan(context)
    expect(context).toBeLessThan(cwd)
  })

  it("renders loading and not-ready residency without changing the model label", () => {
    const loading = renderToStaticMarkup(
      <FooterBar
        context={null}
        model="Qwen Test"
        modelResidency="loading"
        modelLoadingPercentage={42}
      />
    )
    const stopped = renderToStaticMarkup(
      <FooterBar context={null} model="Qwen Test" modelResidency="not-ready" />
    )

    expect(loading).toContain('aria-label="Model loading · 42%"')
    expect(loading).not.toContain("Qwen Test · Loading")
    expect(stopped).toContain('aria-label="Model not ready"')
    expect(stopped).toContain("○")
  })
})
