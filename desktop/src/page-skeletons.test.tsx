import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { ConnectionsSkeleton, HardwareSkeleton, LoginSkeleton, ModelsSkeleton, RecommendationsSkeleton, UpdatesSkeleton } from "./page-skeletons"
import { UsageFigures } from "./serving-usage"
import { MemoryFigures } from "./memory-breakdown"

it("marks loading regions and keeps decorative placeholders out of the accessibility tree", () => {
  for (const component of [<ConnectionsSkeleton />, <HardwareSkeleton />, <LoginSkeleton />, <RecommendationsSkeleton />, <UpdatesSkeleton />, ...(["discover", "catalog", "models"] as const).map(page => <ModelsSkeleton page={page} />)]) {
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
