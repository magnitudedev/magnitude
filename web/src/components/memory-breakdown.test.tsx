import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { Option, Schema } from "effect"
import { ModelInstanceAllocationSchema } from "@magnitudedev/sdk"
import { servingHardware } from "@magnitudedev/client-common/fixtures/hardware"
import { streamingStatus } from "@magnitudedev/client-common/fixtures/model"
import { MemoryFigures } from "./memory-breakdown"

const GiB = 1024 ** 3
const hardware = Option.some(servingHardware("Discrete"))
const allocation = Schema.decodeUnknownSync(ModelInstanceAllocationSchema)({
  contextWindowTokens: 4096,
  memoryDomains: [
    { memoryDomainId: "system", modelBytes: GiB, contextBytes: 0, computeBytes: GiB / 4, auxiliaryBytes: GiB / 4 },
    { memoryDomainId: "cuda-0", modelBytes: 2 * GiB, contextBytes: 2 * GiB, computeBytes: GiB / 4, auxiliaryBytes: GiB / 4 },
  ],
  promptGroupPlaces: { held: 1, capacity: 8 },
})
it("sums the three native allocation categories across memory domains", () => {
  const html = renderToStaticMarkup(<MemoryFigures allocation={Option.some(allocation)} hardware={hardware} />)
  expect(html).toContain(`data-memory-bytes="${6 * GiB}"`)
  for (const [label, bytes] of [["Weights", 3.5 * GiB], ["KV cache", 2 * GiB], ["Runtime", GiB / 2]]) {
    expect(html).toContain(`data-memory-category="${label}" data-bytes="${bytes}"`)
  }
  expect(html.match(/data-memory-category=/g)).toHaveLength(3)
})
it("states how a loaded model serves, and where its bank's experts are only when it has one", () => {
  const streamed = { ...allocation, streaming: Option.some(streamingStatus(["cuda-0", "system"])) }
  const html = renderToStaticMarkup(<MemoryFigures allocation={Option.some(streamed)} hardware={hardware} />)
  expect(html).toContain('aria-label="Serving"')
  expect(html).toContain('data-model-fact="Prompt group places">1 of 8')
  expect(html).toContain('data-model-fact="Experts in memory">40%, floor 1%, ceiling 60%')
  expect(html).toContain('data-model-fact="Experts in Test GPU · GPU 1">300 MB')
  expect(html).toContain('data-model-fact="Experts in System memory">400 MB')
  expect(html).toContain('data-model-fact="Cold experts in System memory">210 MB, 50% of its most')
  expect(html).toContain('data-model-fact="Experts on disk">300 MB')
  expect(html).toContain('data-model-fact="N-gram table in memory">126 MB, 50% of its most, floor 25%')
  expect(html).toContain('data-model-fact="Last resize">Experts shrank from 50% to 40% to make room for a request on ')
  const resident = renderToStaticMarkup(<MemoryFigures allocation={Option.some(allocation)} hardware={hardware} />)
  expect(resident).toContain('data-model-fact="Prompt group places">1 of 8')
  expect(resident).not.toContain("Experts in")
})
it("states prompt group places and the bank's share before the hardware is known, naming no memory", () => {
  const streamed = { ...allocation, streaming: Option.some(streamingStatus(["cuda-0", "system"])) }
  const html = renderToStaticMarkup(<MemoryFigures allocation={Option.some(streamed)} hardware={Option.none()} />)
  expect(html).toContain('data-model-fact="Prompt group places">1 of 8')
  expect(html).toContain('data-model-fact="Experts in memory">40%, floor 1%, ceiling 60%')
  expect(html).not.toContain("Experts in System memory")
  expect(html).not.toContain("Cold experts")
})
it("distinguishes no loaded model from memory that is not available yet", () => {
  const html = renderToStaticMarkup(<MemoryFigures allocation={Option.none()} hardware={hardware} />)
  expect(html).toContain('data-memory-bytes="0">0 MB</strong>')
  expect(html.match(/data-bytes="0"/g)).toHaveLength(3)
  expect(html).not.toContain('NaN')
  for (const text of ['<details', 'System &amp; apps', 'Model buffers', 'processes', 'How memory']) expect(html).not.toContain(text)
  const pending = renderToStaticMarkup(<MemoryFigures allocation={null} hardware={hardware} />)
  expect(pending).toContain("Available once loaded")
  expect(pending).not.toContain("data-memory-bytes")
  expect(pending).not.toContain("data-memory-category")
})
