import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { LocalModelFitResidencySchema, type LocalModel, type StreamingResize } from "@magnitudedev/sdk"
import { servingHardware } from "../application/fixtures/hardware"
import { makeSetupModel, streamedFitResidency, streamedReadyResidency } from "../application/fixtures/model"
import { formatMemoryCapacity, formatMemorySize, formatStorageSize } from "./format-bytes"
import {
  fitResidencyFacts,
  localModelExpertFacts,
  localModelNamesMemory,
  modelServingFacts,
  residentModelAllocation,
  streamingResidencySummary,
} from "./model-streaming"

const allocation = (memoryDomainIds: readonly string[]) =>
  Option.getOrThrow(residentModelAllocation(streamedReadyResidency(memoryDomainIds)))
const discrete = Option.some(servingHardware("Discrete"))
/** The bank on `memoryDomainIds` with only `table`'s and the host tier's last resizes, at those times. */
const withHoldingResizes = (
  memoryDomainIds: readonly string[],
  resizes: { readonly table?: StreamingResize; readonly host?: StreamingResize },
) => {
  const streamed = allocation(memoryDomainIds)
  const status = Option.getOrThrow(streamed.streaming)
  return {
    ...streamed,
    streaming: Option.some({
      ...status,
      lastResize: Option.none(),
      tables: status.tables.map((table) => ({ ...table, lastResize: Option.fromNullable(resizes.table) })),
      host: Option.map(status.host, (host) => ({ ...host, lastResize: Option.fromNullable(resizes.host) })),
    }),
  }
}

/** A model whose fit is streamed (60% allowed), unloaded or loaded with an expert bank (40% held). */
const streamedModel = (loaded: boolean): LocalModel => {
  const model = makeSetupModel(true)
  const serving = model.servingState
  if (serving._tag !== "Assessed" || serving.assessment._tag !== "Fits") throw new Error("the setup model fits")
  return {
    ...model,
    servingState: { ...serving, assessment: { ...serving.assessment, residency: streamedFitResidency() } },
    acquisitionState: loaded
      ? { ...model.acquisitionState, residencyState: streamedReadyResidency(["cuda-0", "system"]) }
      : model.acquisitionState,
  } as LocalModel
}

describe("model serving facts", () => {
  it("states prompt group places, where the bank's experts, cold experts and table are, and its last resize", () => {
    const facts = modelServingFacts(allocation(["cuda-0", "system"]), discrete)
    expect(facts.slice(0, 7)).toEqual([
      ["Prompt group places", "1 of 8"],
      ["Experts in memory", "40%, floor 1%, ceiling 60%"],
      ["Experts in Test GPU · GPU 1", "300 MB"],
      ["Experts in System memory", "400 MB"],
      ["Cold experts in System memory", "210 MB, 50% of its most"],
      ["Experts on disk", "300 MB"],
      ["N-gram table in memory", "126 MB, 50% of its most, floor 25%"],
    ])
    expect(facts[7]?.[0]).toBe("Last resize")
    expect(facts[7]?.[1]).toMatch(/^Experts shrank from 50% to 40% as memory grew short on /)
  })

  it("dates the last resize, so one days ago never reads as today's", () => {
    const facts = modelServingFacts(allocation(["system"]), discrete)
    const day = new Date(1234).toLocaleString([], { month: "short", day: "numeric" })
    expect(facts.at(-1)?.[1]).toContain(` on ${day}`)
  })

  it("names a cold experts resize by its share of its most", () => {
    expect(modelServingFacts(withHoldingResizes(["cuda-0"], {
      host: { _tag: "Grew", from: 15, by: 15, atUnixMs: 1234 },
    }), discrete).at(-1)?.[1]).toMatch(/^Cold experts grew from 25% to 50% into free memory on /)
  })

  it("names the table a resize changed, by its share of its most", () => {
    expect(modelServingFacts(withHoldingResizes(["system"], {
      table: { _tag: "Shrank", to: 20, by: 10, atUnixMs: 1234 },
    }), discrete).at(-1)?.[1]).toMatch(/^N-gram table shrank from 75% to 50% as memory grew short on /)
  })

  it("states the latest of the holdings' last resizes", () => {
    expect(modelServingFacts(withHoldingResizes(["cuda-0"], {
      table: { _tag: "Shrank", to: 20, by: 10, atUnixMs: 1000 },
      host: { _tag: "Grew", from: 15, by: 15, atUnixMs: 2000 },
    }), discrete).at(-1)?.[1]).toMatch(/^Cold experts grew/)
  })

  it("names a tier in memory the GPU shares with the host as unified memory, with no cold experts", () => {
    const facts = modelServingFacts(allocation(["system"]), Option.some(servingHardware("Unified")))
    expect(facts[2]).toEqual(["Experts in Unified memory", "300 MB"])
    expect(facts.map(([label]) => label)).not.toContainEqual(expect.stringMatching(/^Cold experts/))
  })

  it("states places and every hardware-free fact before the hardware is known, and names no memory", () => {
    const labels = modelServingFacts(allocation(["cuda-0", "system"]), Option.none()).map(([label]) => label)
    expect(labels).toEqual([
      "Prompt group places", "Experts in memory", "Experts on disk", "N-gram table in memory", "Last resize",
    ])
  })

  it("refuses a tier whose memory domain the known hardware does not name", () => {
    expect(() => modelServingFacts(allocation(["cuda-7", "system"]), discrete)).toThrow(/cuda-7/)
  })

  it("states places, and nothing of experts, for a load without a bank", () => {
    const resident = { ...allocation(["system"]), streaming: Option.none() }
    expect(modelServingFacts(resident, discrete)).toEqual([["Prompt group places", "1 of 8"]])
  })
})

describe("model expert facts", () => {
  it("states a streamed fit's allowed share, floor and floor speed, and nothing for a resident fit", () => {
    const resident = Schema.decodeUnknownSync(LocalModelFitResidencySchema)({
      _tag: "Resident", performance: { _tag: "Estimated", samples: [{ contextTokens: 32_768, estimatedTokensPerSecond: 40 }] },
    })
    expect(fitResidencyFacts(resident, 32_768)).toEqual([])
    expect(fitResidencyFacts(streamedFitResidency(), 32_768)).toEqual([
      ["Experts this machine allows", "60%, the rest read from disk"],
      ["Expert floor", "1%"],
      ["Speed at the floor", "~5 tok/s"],
    ])
    expect(fitResidencyFacts(streamedFitResidency({ _tag: "HostRam", cacheBytes: 9_000_000_000 }), 32_768)[0])
      .toEqual(["Experts this machine allows", `60%, the rest in system memory as it allows (${formatMemorySize(9_000_000_000)} cached), else read from disk`])
  })

  it("states an unloaded streamed fit's facts, and needs no hardware for them", () => {
    const model = streamedModel(false)
    expect(localModelNamesMemory(model)).toBe(false)
    expect(localModelExpertFacts(model, Option.none())[0]).toEqual(["Experts this machine allows", "60%, the rest read from disk"])
  })

  it("states a loaded streamed model's bank in place of its fit, each label once", () => {
    const model = streamedModel(true)
    expect(localModelNamesMemory(model)).toBe(true)
    const labels = localModelExpertFacts(model, discrete).map(([label]) => label)
    expect(labels[0]).toBe("Experts in memory")
    expect(labels).not.toContain("Experts this machine allows")
    expect(new Set(labels).size).toBe(labels.length)
  })
})

describe("streaming residency summary", () => {
  it("formats unified memory residency with range and in-memory/disk bytes", () => {
    const base = Option.getOrThrow(allocation(["system"]).streaming)
    const status = {
      ...base,
      experts: 100,
      slots: 44,
      floorSlots: 2,
      ceilingSlots: 100,
      tiers: base.tiers.map((tier) => ({ ...tier, bytes: Math.round(15.5 * 1024 ** 3) })),
      diskBytes: 19 * 1024 ** 3,
    }
    expect(streamingResidencySummary(status)).toEqual({
      primary: "44%",
      range: "(2%–100%)",
      secondary: null,
      breakdown: `${formatMemoryCapacity(Math.round(15.5 * 1024 ** 3))} in memory · ${formatStorageSize(19 * 1024 ** 3)} on disk`,
    })
    expect(streamingResidencySummary({ ...status, diskBytes: 0 }).breakdown)
      .toBe(`${formatMemoryCapacity(Math.round(15.5 * 1024 ** 3))} in memory`)
  })

  it("formats discrete 3-tier residency with GPU range, RAM share, and optional disk bytes", () => {
    const base = Option.getOrThrow(allocation(["cuda-0"]).streaming)
    const host = Option.getOrThrow(base.host)
    const status = {
      ...base,
      experts: 100,
      slots: 38,
      floorSlots: 2,
      ceilingSlots: 62,
      tiers: base.tiers.map((tier) => ({ ...tier, bytes: Math.round(12.4 * 1024 ** 3) })),
      host: Option.some({ ...host, slots: 62, ceilingSlots: 62, bytes: Math.round(20.1 * 1024 ** 3) }),
      diskBytes: 0,
    }
    expect(streamingResidencySummary(status)).toEqual({
      primary: "38% GPU",
      range: "(2%–62%)",
      secondary: "62% RAM",
      breakdown: `${formatMemoryCapacity(Math.round(12.4 * 1024 ** 3))} GPU · ${formatMemoryCapacity(Math.round(20.1 * 1024 ** 3))} RAM`,
    })
    expect(streamingResidencySummary({ ...status, diskBytes: 5 * 1024 ** 3 }).breakdown)
      .toBe(`${formatMemoryCapacity(Math.round(12.4 * 1024 ** 3))} GPU · ${formatMemoryCapacity(Math.round(20.1 * 1024 ** 3))} RAM · ${formatStorageSize(5 * 1024 ** 3)} on disk`)
  })

  it("excludes the host memory domain tier from GPU bytes instead of double-counting RAM into GPU", () => {
    const base = Option.getOrThrow(allocation(["cuda-0", "system"]).streaming)
    const host = Option.getOrThrow(base.host)
    const gpuBytes = 20 * 1024 ** 3
    const ramBytes = 22 * 1024 ** 3
    const diskBytes = 16_300_000_000
    const status = {
      ...base,
      experts: 100,
      slots: 88,
      floorSlots: 0,
      ceilingSlots: 96,
      tiers: [
        { memoryDomainId: base.tiers[0]!.memoryDomainId, bytes: gpuBytes },
        { memoryDomainId: host.memoryDomainId, bytes: ramBytes },
      ],
      host: Option.some({ ...host, slots: 100, ceilingSlots: 100, bytes: ramBytes }),
      diskBytes,
    }
    expect(streamingResidencySummary(status)).toEqual({
      primary: "88% GPU",
      range: "(0%–96%)",
      secondary: "100% RAM",
      breakdown: "20 GB GPU · 22 GB RAM · 16.3 GB on disk",
    })
  })
})
