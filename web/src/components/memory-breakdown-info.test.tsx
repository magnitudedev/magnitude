import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { MemoryBreakdownSummary } from "./memory-breakdown-info"
import { fits } from "../../test/models/fixtures"

const GB = 1_000_000_000
const text = (html: string) => html.replace(/<[^>]+>/g, "")

describe("memory breakdown", () => {
  it("states the loaded total and each part, with the context at its length", () => {
    const html = text(renderToStaticMarkup(<MemoryBreakdownSummary assessment={{
      ...fits,
      profile: { contextLength: 1_048_576 },
      served: { contextLength: 1_048_576 },
      memory: { ...fits.memory, totalRequiredBytes: 45.4 * GB, breakdown: { weightsBytes: 36 * GB, contextBytes: 7.1 * GB, overheadBytes: 2.3 * GB } },
    }} />))
    expect(html).toContain("This model takes up 45.4 GB when loaded")
    expect(html).toContain("Weights 36 GB")
    expect(html).toContain("1M context 7.1 GB")
    expect(html).toContain("Overhead 2.3 GB")
    expect(html).not.toContain("doesn’t fit")
  })

  it("says the full context doesn't fit when the model is served below it", () => {
    const html = text(renderToStaticMarkup(<MemoryBreakdownSummary assessment={{
      ...fits,
      profile: { contextLength: 1_048_576 },
      served: { contextLength: 262_144 },
    }} />))
    expect(html).toContain("256K context 4 GB")
    expect(html).toContain("(full 1M doesn’t fit)")
  })
})
