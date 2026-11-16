import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import type { ServingUsageSnapshot } from "@magnitudedev/sdk"
import { ApiEquivalentCost, formatCostCurrency, formatExactCost } from "./api-equivalent-cost"

const sampleUsage: Extract<ServingUsageSnapshot, { _tag: "Available" }> = {
  _tag: "Available",
  since: 1700000000000,
  dailyActivity: [],
  requests: 10,
  incompleteRequests: 0,
  inputTokens: 100_000,
  cachedInputTokens: 40_000,
  outputTokens: 20_000,
  totalTokens: 120_000,
  cachedInputRequests: 10,
  tokensPerSecond: 65,
  timeToFirstTokenMs: 150,
  speedSamples: 10,
  latencySamples: 10,
  models: [],
  recordingFailures: 0,
}

describe("ApiEquivalentCost Component", () => {
  it("formats currency values accurately with sub-cent and exact formatting", () => {
    expect(formatCostCurrency(0)).toBe("$0.00")
    expect(formatCostCurrency(0.004)).toBe("< $0.01")
    expect(formatCostCurrency(1.25)).toBe("$1.25")
    expect(formatExactCost(0.0042)).toBe("$0.0042")
    expect(formatExactCost(0)).toBe("$0.0000")
  })

  it("renders cost summary and breakdown without claiming net savings", () => {
    // Claude 3.5 Sonnet pricing:
    // Uncached: 60,000 * $3.00/1M = $0.18
    // Cached: 40,000 * $0.30/1M = $0.012
    // Output: 20,000 * $15.00/1M = $0.30
    // Total = $0.492 -> "$0.49"
    const html = renderToStaticMarkup(
      <ApiEquivalentCost
        usage={sampleUsage}
        selectedModelId="claude-3-5-sonnet-20241022"
        selectedModelDisplayName="Claude 3.5 Sonnet"
        period="Today"
      />,
    )

    expect(html).toContain('API-Equivalent Cost')
    expect(html).toContain('data-cost="total"')
    expect(html).toContain('$0.49')
    expect(html).toContain('data-cost="uncached-input"')
    expect(html).toContain('$0.18')
    expect(html).toContain('data-cost="cached-input"')
    expect(html).toContain('$0.01')
    expect(html).toContain('data-cost="output"')
    expect(html).toContain('$0.30')
    expect(html).toContain('Published rates')
    // Must contain the exact required transparency disclaimer
    expect(html).toMatch(
      /API-equivalent cost estimates what your local inference usage would cost at the selected hosted provider(&#x27;|')s published token rates\. This is not your actual expenditure or net savings\./,
    )
    expect(html).not.toContain("actual savings")
    expect(html).not.toContain("net savings achieved")
  })

  it("renders explicit unmapped state for unknown custom local models", () => {
    const html = renderToStaticMarkup(
      <ApiEquivalentCost
        usage={sampleUsage}
        selectedModelId="custom-unmapped-finetune:gguf:q4"
        selectedModelDisplayName="My Custom Fine-Tune"
        period="Today"
      />,
    )

    expect(html).toContain("No default hosted equivalent for this model")
    expect(html).toContain("Select a comparable model from the dropdown above")
    expect(html).not.toContain('data-cost="total"')
  })

  it("renders skeleton placeholders when usage is loading without inventing zero costs", () => {
    const html = renderToStaticMarkup(
      <ApiEquivalentCost
        usage={null}
        selectedModelId="claude-3-5-sonnet-20241022"
        period="Today"
      />,
    )

    expect(html).toContain('data-slot="skeleton"')
    expect(html).not.toContain('data-cost="total"')
    expect(html).not.toContain('$0.00')
  })

  it("handles zero token requests accurately as $0.00", () => {
    const zeroUsage = {
      ...sampleUsage,
      inputTokens: 0,
      cachedInputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      requests: 0,
    }

    const html = renderToStaticMarkup(
      <ApiEquivalentCost
        usage={zeroUsage}
        selectedModelId="claude-3-5-sonnet-20241022"
        period="Today"
      />,
    )

    expect(html).toContain('data-cost="total"')
    expect(html).toContain('$0.00')
  })

  it("exposes comparison model options in the select dropdown", () => {
    const html = renderToStaticMarkup(
      <ApiEquivalentCost
        usage={sampleUsage}
        selectedModelId="claude-3-5-sonnet-20241022"
        period="Today"
      />,
    )

    expect(html).toContain("Compare against:")
    expect(html).toContain("Auto (Claude 3.5 Sonnet)")
  })
})
