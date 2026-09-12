import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { Option, Schema } from "effect"
import { ModelInstanceAllocationSchema } from "@magnitudedev/sdk"
import { MemoryFigures } from "./memory-breakdown"

const GiB = 1024 ** 3
const allocation = Schema.decodeUnknownSync(ModelInstanceAllocationSchema)({
  contextWindowTokens: 4096, parallelSequences: 1, physicalContextTokens: 4096,
  memoryDomains: [
    { memoryDomainId: "system", modelBytes: GiB, contextBytes: 0, computeBytes: GiB / 4, auxiliaryBytes: GiB / 4 },
    { memoryDomainId: "gpu", modelBytes: 2 * GiB, contextBytes: 2 * GiB, computeBytes: GiB / 4, auxiliaryBytes: GiB / 4 },
  ],
})
it("sums the three native allocation categories across memory domains", () => {
  const html = renderToStaticMarkup(<MemoryFigures allocation={Option.some(allocation)} />)
  expect(html).toContain(`data-memory-bytes="${6 * GiB}"`)
  for (const [label, bytes] of [["Model weights", 3 * GiB], ["KV cache", 2 * GiB], ["Overhead", GiB]]) {
    expect(html).toContain(`data-memory-category="${label}" data-bytes="${bytes}"`)
  }
  expect(html.match(/data-memory-category=/g)).toHaveLength(3)
})
it("shows zero total and zero categories with no loaded model", () => {
  const html = renderToStaticMarkup(<MemoryFigures allocation={Option.none()} />)
  expect(html).toContain('data-memory-bytes="0">0 MB</p>')
  expect(html.match(/data-bytes="0"/g)).toHaveLength(3)
  expect(html).not.toContain('NaN')
  for (const text of ['<details', 'System &amp; apps', 'Model buffers', 'processes', 'How memory']) expect(html).not.toContain(text)
})
