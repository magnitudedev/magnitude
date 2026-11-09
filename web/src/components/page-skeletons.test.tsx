import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { Option } from "effect"
import { ConnectionsSkeleton, HardwarePending, LoginSkeleton, ModelsSkeleton, UpdatesSkeleton } from "./page-skeletons"
import { UsageFigures } from "./serving-usage"
import { MemoryFigures } from "./memory-breakdown"

it("marks loading regions and keeps decorative placeholders out of the accessibility tree", () => {
  for (const component of [<ConnectionsSkeleton />, <LoginSkeleton />, <UpdatesSkeleton />, <ModelsSkeleton />]) {
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
  const memory = renderToStaticMarkup(<MemoryFigures allocation={null} hardware={Option.none()} reading />)
  expect(memory).toContain('data-slot="skeleton"')
  expect(memory).not.toContain('0 MB')
  expect(memory).not.toContain('data-memory-bytes="0"')
  expect(memory).toContain('Model memory')
  expect(memory).not.toContain('data-memory-category')
})

it("reports hardware detection and service start inline without inventing progress", () => {
  expect(renderToStaticMarkup(<HardwarePending identifying />)).toContain("Identifying your machine")
  const hardware = renderToStaticMarkup(<HardwarePending />)
  expect(hardware).toContain('role="status"')
  expect(hardware).toContain("Reading hardware capabilities")
  expect(hardware).not.toContain("<figure")
  const starting = renderToStaticMarkup(<ModelsSkeleton startingService />)
  expect(starting).toContain("Models</h1>")
  expect(starting).toContain("Starting Magnitude…")
  expect(starting).not.toContain("<progress")
})
