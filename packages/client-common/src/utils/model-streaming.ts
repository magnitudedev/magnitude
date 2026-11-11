import { Array, Option } from "effect"
import {
  localModelResidency,
  localModelServingState,
  type ColdExperts,
  type LocalInferenceHardware,
  type LocalInferenceMemoryDomainId,
  type LocalModel,
  type LocalModelFitResidency,
  type ModelInstanceAllocation,
  type ModelResidency,
  type ModelStreamingStatus,
  type StreamingResize,
} from "@magnitudedev/sdk"
import { formatMemorySize } from "./format-bytes"
import { memoryDomainLabels } from "./hardware-memory"
import { performanceRangeSpeedLabel } from "./local-model-radar"

/** One fact as the status surfaces print it: a label and its value. */
export type ModelFact = readonly [label: string, value: string]

/** The allocation of a load that holds its memory: ready, or stopping after it was ready. */
export const residentModelAllocation = (residency: ModelResidency): Option.Option<ModelInstanceAllocation> => {
  if (residency._tag === "Ready") return Option.some(residency.allocation)
  if (residency._tag === "Stopping" && residency.allocation._tag === "Resident") {
    return Option.some(residency.allocation.allocation)
  }
  return Option.none()
}

/** The allocation a loaded model holds (`residentModelAllocation`); none for a model not loaded. */
export const localModelAllocation = (model: LocalModel): Option.Option<ModelInstanceAllocation> =>
  Option.flatMap(localModelResidency(model), residentModelAllocation)

const percent = (share: number): string => `${Math.round(share * 100)}%`

const COLD_EXPERTS: Record<ColdExperts, string> = {
  disk: "the rest read from disk",
  host_ram: "the rest in system memory as it allows, else read from disk",
}

/**
 * A streamed fit's facts: the share of its experts this machine allows in memory and where the rest
 * are kept, the floor it never goes below, and its speed there. None for a resident fit.
 */
export const fitResidencyFacts = (
  residency: LocalModelFitResidency,
  contextLength: number,
): readonly ModelFact[] => {
  if (residency._tag === "Resident") return []
  const { bank } = residency
  return [
    ["Experts this machine allows", `${percent(bank.slots / bank.experts)}, ${COLD_EXPERTS[bank.coldExperts]}`],
    ["Expert floor", percent(bank.floorSlots / bank.experts)],
    ["Speed at the floor", performanceRangeSpeedLabel(residency.floorPerformance, contextLength)],
  ]
}

/** A fitting model's streamed-fit facts; none for a resident fit or a model that does not fit. */
export const localModelFitFacts = (model: LocalModel): readonly ModelFact[] => {
  const serving = Option.getOrUndefined(localModelServingState(model))
  return serving?._tag === "Assessed" && serving.assessment._tag === "Fits"
    ? fitResidencyFacts(serving.assessment.residency, serving.assessment.served.contextLength)
    : []
}

/** A moment with its date, so a resize days ago never reads as today's. */
const dateTime = (unixMs: number): string =>
  new Date(unixMs).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })

/** A table's label among the bank's: "N-gram table", numbered when there are several. */
const tableName = (status: ModelStreamingStatus, table: number): string =>
  status.tables.length > 1 ? `N-gram table ${table + 1}` : "N-gram table"

/** A holding's last resize, with its name and how its sizes read: a share of what it is a share of. */
interface HoldingResize {
  readonly name: string
  readonly share: (size: number) => string
  readonly resize: StreamingResize
}

/** A holding's resize as a sentence: the verb, its sizes before and after, and why. */
const resizeSentence = ({ name, share, resize }: HoldingResize): string => resize._tag === "Shrank"
  ? `${name} shrank from ${share(resize.to + resize.by)} to ${share(resize.to)} as memory grew short`
  : `${name} grew from ${share(resize.from)} to ${share(resize.from + resize.by)} into free memory`

/**
 * The bank's latest resize, of whichever holding it changed: its experts (a share of every expert), a
 * table (a share of its ceiling) or the cold experts (the host tier's share of its ceiling).
 */
const latestResize = (status: ModelStreamingStatus): Option.Option<HoldingResize> => {
  const resized = (name: string, share: (size: number) => string, resize: Option.Option<StreamingResize>) =>
    Option.map(resize, (last): HoldingResize => ({ name, share, resize: last }))
  const holdings = Array.getSomes([
    resized("Experts", (slots) => percent(slots / status.experts), status.lastResize),
    ...status.tables.map((table, index) =>
      resized(tableName(status, index), (records) => percent(records / table.ceilingRecords), table.lastResize)),
    Option.flatMap(status.host, (host) =>
      resized("Cold experts", (slots) => percent(slots / host.ceilingSlots), host.lastResize)),
  ])
  return Array.reduce(holdings, Option.none<HoldingResize>(), (latest, holding) =>
    Option.exists(latest, ({ resize }) => resize.atUnixMs >= holding.resize.atUnixMs) ? latest : Option.some(holding))
}

/**
 * Where an expert bank's experts are, as facts: the share of every expert it holds, with its floor
 * and ceiling; its bytes in each memory tier and, on a discrete device, the cold experts the host
 * tier holds beside it, each in the memory domain the hardware names (stated once the hardware is
 * known); the bytes left on disk; each table it holds beside them, its share of its ceiling and
 * its bytes; and the last resize, of whichever it changed. Known hardware names every memory
 * domain a load holds memory in: one it does not name is the service's defect, and throws.
 */
const expertBankFacts = (
  status: ModelStreamingStatus,
  hardware: Option.Option<LocalInferenceHardware>,
): readonly ModelFact[] => {
  const labels = Option.map(hardware, memoryDomainLabels)
  const named = (id: LocalInferenceMemoryDomainId) => Option.map(labels, (names) => {
    const name = names.get(id)
    if (name === undefined) throw new Error(`The hardware names no memory domain ${id}, which an expert bank holds memory in`)
    return name
  })
  const share = (slots: number) => percent(slots / status.experts)
  return [
    ["Experts in memory", `${share(status.slots)}, floor ${share(status.floorSlots)}, ceiling ${share(status.ceilingSlots)}`],
    ...status.tiers.flatMap((tier): readonly ModelFact[] =>
      Option.match(named(tier.memoryDomainId), { onNone: () => [], onSome: (name) => [[`Experts in ${name}`, formatMemorySize(tier.bytes)]] })),
    ...Option.match(Option.flatMap(status.host, (host) => Option.map(named(host.memoryDomainId), (name) => ({ host, name }))), {
      onNone: () => [],
      onSome: ({ host, name }): readonly ModelFact[] => [[
        `Cold experts in ${name}`,
        `${formatMemorySize(host.bytes)}, ${percent(host.slots / host.ceilingSlots)} of its most`,
      ]],
    }),
    ["Experts on disk", formatMemorySize(status.diskBytes)],
    ...status.tables.map((table, index): ModelFact => [
      `${tableName(status, index)} in memory`,
      `${formatMemorySize(table.bytes)}, ${percent(table.records / table.ceilingRecords)} of its most, floor ${
        percent(table.floorRecords / table.ceilingRecords)}`,
    ]),
    ...Option.match(latestResize(status), {
      onNone: () => [],
      onSome: (latest): readonly ModelFact[] =>
        [["Last resize", `${resizeSentence(latest)} on ${dateTime(latest.resize.atUnixMs)}`]],
    }),
  ]
}

/** The places granted to a loaded model's prompt groups, of the most they may take. */
export const promptGroupPlacesFact = (allocation: ModelInstanceAllocation): ModelFact =>
  ["Prompt group places", `${allocation.promptGroupPlaces.held} of ${allocation.promptGroupPlaces.capacity}`]

/**
 * How a load serves, as facts: the places granted to its prompt groups, and where its expert bank's
 * experts are, named as the hardware (once known) names its memory.
 */
export const modelServingFacts = (
  allocation: ModelInstanceAllocation,
  hardware: Option.Option<LocalInferenceHardware>,
): readonly ModelFact[] => [
  promptGroupPlacesFact(allocation),
  ...Option.match(allocation.streaming, { onNone: () => [], onSome: (status) => expertBankFacts(status, hardware) }),
]

/** The expert bank a loaded model holds; none for a model not loaded, or loaded without one. */
const localModelExpertBank = (model: LocalModel): Option.Option<ModelStreamingStatus> =>
  Option.flatMap(localModelAllocation(model), ({ streaming }) => streaming)

/**
 * What a model's surface states about its experts: its expert bank's facts where a load holds one,
 * and otherwise its streamed fit's, so the bank now and the share the fit allows never both show.
 */
export const localModelExpertFacts = (
  model: LocalModel,
  hardware: Option.Option<LocalInferenceHardware>,
): readonly ModelFact[] => Option.match(localModelExpertBank(model), {
  onNone: () => localModelFitFacts(model),
  onSome: (status) => expertBankFacts(status, hardware),
})

/** Whether a model's expert facts name its memory, so its surface needs the hardware to state them. */
export const localModelNamesMemory = (model: LocalModel): boolean => Option.isSome(localModelExpertBank(model))
