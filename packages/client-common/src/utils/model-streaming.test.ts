import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { LocalModelFitResidencySchema, type LocalModel, type StreamingResize } from "@magnitudedev/sdk"
import { servingHardware } from "../application/fixtures/hardware"
import { makeSetupModel, streamedFitResidency, streamedReadyResidency } from "../application/fixtures/model"
import { formatMemorySize } from "./format-bytes"
import {
  fitResidencyFacts,
  localModelExpertFacts,
  localModelNamesMemory,
  modelServingFacts,
  residentModelAllocation,
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
