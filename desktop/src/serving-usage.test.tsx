import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import type { ServingUsageSnapshot } from "@magnitudedev/sdk"
import { UsageFigures } from "./serving-usage"
const usage: Extract<ServingUsageSnapshot, { _tag: "Available" }> = {
  _tag: "Available", dailyActivity: [], since: 1789187320947, requests: 2, incompleteRequests: 0,
  inputTokens: 100, cachedInputTokens: 40, outputTokens: 20, totalTokens: 120,
  cachedInputRequests: 2, tokensPerSecond: 80, timeToFirstTokenMs: 125,
  speedSamples: 2, latencySamples: 2, models: [], recordingFailures: 0,
}
it("renders all token categories and measurements without counting cache twice", () => {
  const html = renderToStaticMarkup(<UsageFigures usage={usage} />)
  expect(html).not.toContain('data-usage="total"')
  expect(html).not.toContain("total tokens ·")
  for (const [label, count] of [["Input tokens",100],["Cached input",40],["Output tokens",20]]) expect(html).toContain(`data-usage="${label}">${count}</p>`)
  expect(html).toContain("80 tokens/s"); expect(html).toContain("125 ms")
})
it("renders missing measurements without diagnostic notices", () => {
  const html = renderToStaticMarkup(<UsageFigures usage={{ ...usage, cachedInputRequests: 0, tokensPerSecond: null, timeToFirstTokenMs: null, incompleteRequests: 1, recordingFailures: 1 }} />)
  expect(html).toContain('data-usage="Cached input">—</p>')
  expect(html).toContain('data-usage="speed">—</p>'); expect(html).toContain('data-usage="ttft">—</p>')
  expect(html).not.toContain('Totals are partial.')
  expect(html).not.toContain('complete usage evidence')
  expect(html).not.toContain('could not be saved')
})

it("shows requests and cache share without the explanatory section", () => {
  const html = renderToStaticMarkup(<UsageFigures usage={usage} />)
  expect(html).toContain('data-usage="requests">2</p>')
  expect(html).toContain('40% of input tokens served from cache.')
  expect(html).not.toContain('About these numbers')
})
it("does not present missing cache measurements as a zero percentage", () => {
  const html = renderToStaticMarkup(<UsageFigures usage={{ ...usage, cachedInputRequests: 0 }} />)
  expect(html).toContain('Cache usage was not reported.')
  expect(html).not.toContain('% of input tokens served from cache.')
})
