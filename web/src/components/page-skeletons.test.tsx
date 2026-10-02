import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { ConnectionsSkeleton, HardwarePending, LoginSkeleton, ModelsSkeleton, RecommendationsSkeleton, UpdatesSkeleton } from "./page-skeletons"
import { UsageFigures } from "./serving-usage"
import { MemoryFigures } from "./memory-breakdown"

it("marks loading regions and keeps decorative placeholders out of the accessibility tree", () => {
  for (const component of [<HardwarePending />, <ConnectionsSkeleton />, <LoginSkeleton />, <RecommendationsSkeleton />, <UpdatesSkeleton />, ...(["catalog", "models"] as const).map(page => <ModelsSkeleton page={page} />)]) {
    const html = renderToStaticMarkup(component)
    expect(html).toContain('aria-busy="true"')
    expect(html).toContain('role="status"')
    expect(html).toContain('aria-hidden="true"')
    expect(html).not.toMatch(/<(button|input|a)\b/)
    expect(html).toContain('motion-safe:animate-pulse')
  }
})

it("does not turn unobserved usage or memory into zero or an empty-state claim", () => {
  const usage = renderToStaticMarkup(<UsageFigures usage={null} />)
  expect(usage).toContain('data-slot="skeleton"')
  expect(usage).not.toContain('No requests')
  expect(usage).not.toContain('>0<')
  const memory = renderToStaticMarkup(<MemoryFigures allocation={null} />)
  expect(memory).toContain('data-slot="skeleton"')
  expect(memory).not.toContain('0 MB')
  expect(memory).not.toContain('data-memory-bytes="0"')
  for (const category of ['Model weights', 'KV cache', 'Overhead']) expect(memory).toContain(category)
})

it("shows assessment progress inside the recommendation skeleton", () => {
  const html = renderToStaticMarkup(<RecommendationsSkeleton assessment={{ settledModels: 12, totalModels: 55 }} />)
  expect(html).toContain("Assessing models")
  expect(html).toContain("12 of 55 assessed")
  expect(html).toContain('role="status"')
})

it("distinguishes waiting from measurable assessment without inventing progress", () => {
  const waiting = renderToStaticMarkup(<RecommendationsSkeleton waitingForHardware assessment={{ settledModels: 0, totalModels: 55 }} />)
  expect(waiting).toContain("Waiting for hardware")
  expect(waiting).not.toContain('<progress')
  const unknown = renderToStaticMarkup(<RecommendationsSkeleton />)
  expect(unknown).toContain("Loading model catalog")
  expect(unknown).not.toContain('<progress')
  const measured = renderToStaticMarkup(<RecommendationsSkeleton assessment={{ settledModels: 12, totalModels: 55 }} />)
  expect(measured).toContain('max="55" value="12"')
  expect(renderToStaticMarkup(<HardwarePending identifying />)).toContain("Identifying your machine")
  expect(renderToStaticMarkup(<HardwarePending />)).toContain("Reading hardware capabilities")
})
