import { renderToStaticMarkup } from "react-dom/server"
import { expect, it } from "vitest"
import { Option, Schema } from "effect"
import { ModelInstanceAllocationSchema } from "@magnitudedev/sdk"
import { formatMemoryCapacity } from "@magnitudedev/client-common"
import { servingHardware } from "@magnitudedev/client-common/fixtures/hardware"
import { streamingStatus } from "@magnitudedev/client-common/fixtures/model"
import { MemoryFigures } from "./memory-breakdown"
import { MachineCard } from "./models/parts"
import { machineView } from "./models/machine"

const GiB = 1024 ** 3
const discreteHardware = Option.some(servingHardware("Discrete"))
const unifiedHardware = Option.some(servingHardware("Unified"))

const splitAllocation = Schema.decodeUnknownSync(ModelInstanceAllocationSchema)({
  contextWindowTokens: 4096,
  memoryDomains: [
    { memoryDomainId: "system", modelBytes: GiB, contextBytes: 0, computeBytes: GiB / 4, auxiliaryBytes: GiB / 4 },
    { memoryDomainId: "cuda-0", modelBytes: 2 * GiB, contextBytes: 2 * GiB, computeBytes: GiB / 4, auxiliaryBytes: GiB / 4 },
  ],
  promptGroupPlaces: { held: 1, capacity: 8 },
})

const unifiedAllocation = Schema.decodeUnknownSync(ModelInstanceAllocationSchema)({
  contextWindowTokens: 4096,
  memoryDomains: [
    { memoryDomainId: "system", modelBytes: 3 * GiB, contextBytes: 2 * GiB, computeBytes: GiB / 2, auxiliaryBytes: GiB / 2 },
  ],
  promptGroupPlaces: { held: 1, capacity: 8 },
})

it("renders one full-capacity bar on unified memory with model left, free in the middle, and other apps right", () => {
  const html = renderToStaticMarkup(<MemoryFigures allocation={Option.some(unifiedAllocation)} hardware={unifiedHardware} />)
  expect(html.match(/data-memory-domain=/g)).toHaveLength(1)
  expect(html).toContain('data-memory-domain="system"')
  expect(html).toContain("Unified memory")
  expect(html).toContain(`data-memory-bytes="${6 * GiB}"`)
  expect(html).toContain(`model · ${formatMemoryCapacity(16 * GiB)} total`)
  expect(html).toContain("model · 16 GB total")
  for (const [label, bytes] of [
    ["Weights", 3.5 * GiB],
    ["Context", 2 * GiB],
    ["Overhead", 0.5 * GiB],
    ["Free", 8 * GiB],
    ["Other apps", 2 * GiB],
  ]) {
    expect(html).toContain(`data-memory-category="${label}" data-bytes="${bytes}"`)
  }
  expect(html).toContain('class="settings-memory-segment other"')
  expect(html).not.toContain("data-streaming-summary")
  for (const text of ["<details", "Prompt group places", "N-gram table", "Last resize"]) {
    expect(html).not.toContain(text)
  }
})

it("renders stacked full-capacity bars for GPU memory and system memory on discrete hosts", () => {
  const html = renderToStaticMarkup(<MemoryFigures allocation={Option.some(splitAllocation)} hardware={discreteHardware} />)
  expect(html.match(/data-memory-domain=/g)).toHaveLength(2)
  expect(html.indexOf('data-memory-domain="cuda-0"')).toBeLessThan(html.indexOf('data-memory-domain="system"'))
  expect(html).toContain("Test GPU · GPU 1")
  expect(html).toContain("System memory")
  expect(html).toContain(`data-memory-bytes="${4.5 * GiB}"`)
  expect(html).toContain(`data-memory-bytes="${1.5 * GiB}"`)
})

it("matches MachineCard hardware capacity units for GPU and RAM totals", () => {
  const base = servingHardware("Discrete")
  const hw = {
    ...base,
    totalSystemMemoryBytes: 32 * GiB,
    memoryDomains: base.memoryDomains.map(domain => domain.kind === "PhysicalDevice"
      ? { ...domain, totalBytes: Math.round(8.5 * GiB), availableBytes: Option.none() }
      : { ...domain, totalBytes: 32 * GiB, availableBytes: Option.some(24 * GiB) }),
  }
  const machineHtml = renderToStaticMarkup(<MachineCard machine={machineView(hw)} />)
  const statusHtml = renderToStaticMarkup(<MemoryFigures allocation={Option.some(splitAllocation)} hardware={Option.some(hw)} />)
  expect(machineHtml).toContain(">8.5 GB<")
  expect(machineHtml).toContain(">32 GB<")
  expect(statusHtml).toContain("8.5 GB total")
  expect(statusHtml).toContain("32 GB total")
  expect(statusHtml).not.toContain("9.1 GB total")
  expect(statusHtml).not.toContain("34.4 GB total")
})

it("renders a single compact streaming residency row for streamed models and nothing for resident models", () => {
  const unifiedStreamed = { ...unifiedAllocation, streaming: Option.some(streamingStatus(["system"])) }
  const unifiedHtml = renderToStaticMarkup(<MemoryFigures allocation={Option.some(unifiedStreamed)} hardware={unifiedHardware} />)
  expect(unifiedHtml).toContain("data-streaming-summary")
  expect(unifiedHtml).toContain("Experts in memory")
  expect(unifiedHtml).toContain("<strong>40%</strong>")
  expect(unifiedHtml).toContain("(1%–60%)")
  expect(unifiedHtml).toContain("286 MB in memory · 300 MB on disk")
  for (const text of ["<details", "Prompt group places", "N-gram table", "Last resize"]) {
    expect(unifiedHtml).not.toContain(text)
  }

  const discreteStreamed = { ...splitAllocation, streaming: Option.some(streamingStatus(["cuda-0"])) }
  const discreteHtml = renderToStaticMarkup(<MemoryFigures allocation={Option.some(discreteStreamed)} hardware={discreteHardware} />)
  expect(discreteHtml).toContain("<strong>40% GPU</strong>")
  expect(discreteHtml).toContain("(1%–60%)")
  expect(discreteHtml).toContain("· 30% RAM")
  expect(discreteHtml).toContain("286 MB GPU · 200 MB RAM · 300 MB on disk")
  for (const text of ["<details", "Prompt group places", "N-gram table", "Last resize"]) {
    expect(discreteHtml).not.toContain(text)
  }

  const twoTierBase = streamingStatus(["cuda-0", "system"])
  const twoTierHost = Option.getOrThrow(twoTierBase.host)
  const twoTierStatus = {
    ...twoTierBase,
    tiers: [
      { memoryDomainId: twoTierBase.tiers[0]!.memoryDomainId, bytes: 20 * GiB },
      { memoryDomainId: twoTierHost.memoryDomainId, bytes: 22 * GiB },
    ],
    host: Option.some({ ...twoTierHost, bytes: 22 * GiB }),
    diskBytes: 16_300_000_000,
  }
  const twoTierHtml = renderToStaticMarkup(
    <MemoryFigures allocation={Option.some({ ...splitAllocation, streaming: Option.some(twoTierStatus) })} hardware={discreteHardware} />,
  )
  expect(twoTierHtml).toContain("20 GB GPU · 22 GB RAM · 16.3 GB on disk")
  expect(twoTierHtml).not.toContain("42 GB GPU")
})

it("falls back gracefully when free memory or hardware is unobserved", () => {
  const missingFreeHardware = Option.some({
    ...servingHardware("Unified"),
    memoryDomains: servingHardware("Unified").memoryDomains.map(domain => ({ ...domain, availableBytes: Option.none() })),
  })
  const missingFreeHtml = renderToStaticMarkup(<MemoryFigures allocation={Option.some(unifiedAllocation)} hardware={missingFreeHardware} />)
  expect(missingFreeHtml).toContain(`data-memory-category="Weights" data-bytes="${3.5 * GiB}"`)
  expect(missingFreeHtml).not.toContain('data-memory-category="Free"')
  expect(missingFreeHtml).not.toContain('data-memory-category="Other apps"')
  expect(missingFreeHtml).toContain("Current memory usage is unavailable.")

  const noHardwareHtml = renderToStaticMarkup(<MemoryFigures allocation={Option.some(splitAllocation)} hardware={Option.none()} />)
  expect(noHardwareHtml).toContain(`data-memory-bytes="${6 * GiB}"`)
  expect(noHardwareHtml).not.toContain('data-memory-category="Free"')
})

it("distinguishes no loaded model from memory that is not available yet", () => {
  const unloaded = renderToStaticMarkup(<MemoryFigures allocation={Option.none()} hardware={unifiedHardware} />)
  expect(unloaded).toContain('data-memory-bytes="0">0 MB</strong>')
  expect(unloaded).toContain('data-memory-category="Free"')
  expect(unloaded).toContain('data-memory-category="Other apps"')
  expect(unloaded).not.toContain("NaN")
  expect(unloaded).not.toContain("<details")
  expect(unloaded).not.toContain("data-streaming-summary")

  const pending = renderToStaticMarkup(<MemoryFigures allocation={null} hardware={unifiedHardware} />)
  expect(pending).toContain("Available once loaded")
  expect(pending).not.toContain("data-memory-bytes")
  expect(pending).not.toContain("data-memory-category")
})
