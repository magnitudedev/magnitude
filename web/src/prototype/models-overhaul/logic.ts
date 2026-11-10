import type { Machine, Model, Quant, Repo } from "./data"
import { pool } from "./data"

export type Placement =
  | { readonly _tag: "Resident"; readonly tier: "GPU" | "Memory" }
  | { readonly _tag: "Streamed"; readonly from: "RAM" | "SSD" }
  | { readonly _tag: "DoesNotFit" }

export interface Assessment {
  readonly quant: Quant
  readonly label: QualityLabel
  readonly placement: Placement
  readonly speed: number
  readonly tiers: readonly { readonly name: string; readonly usedGB: number; readonly totalGB: number | null }[]
  readonly diskShortGB: number
  readonly servedK: number
}

export type QualityLabel = string

export const TARGET_SPEED = 20
export const ONBOARDING_FLOOR = 10

/** A quant's nominal bit width, read from its type name: what people already know quants by. */
export const quantBits = (name: string): number => {
  const upper = name.toUpperCase()
  if (/BF16|F16/.test(upper)) return 16
  const digit = upper.match(/(?:IQ|TQ|Q)(\d)/)
  if (digit) return Number(digit[1])
  return /FP4/.test(upper) ? 4 : 0
}

export const qualityLabel = (_model: Model, quant: Quant): QualityLabel => `${quantBits(quant.name)}-bit`

/** For quants Magnitude has measured; a stand-in for the measured KL divergence. */
export const measuredQuality = (quant: Quant) => {
  const bits = quantBits(quant.name)
  return bits >= 6 ? "Matches the original" : bits >= 4 ? "Close to the original" : bits >= 3 ? "Slightly less accurate" : "Less accurate"
}

const efficiency = { moe: 0.6, dense: 0.7 }
const stepOverhead = 0.004
const missCurve = (share: number) => 0.204 * Math.pow(Math.max(0, 1 - share), 2.38)

// Rough decode model at 64K context, calibrated to engine assessments on an M4 Max 64 GB:
// always-used weights stay put, the experts that fit sit in the fastest tier, misses are read from the next.
/** The tiers a model can be served at: its trained limit, then every power of two from 32K below it. */
const contextTiers = (limitK: number) => [limitK, ...[1024, 512, 256, 128, 64, 32].filter(tier => tier < limitK)]

/** The largest tier whose context fits beside what must stay in fast memory; 0 when even 32K doesn't. */
const servedContext = (model: Model, fixedGB: number, roomGB: number) =>
  contextTiers(model.contextK).find(tier => fixedGB + tier * model.kvPerK <= roomGB) ?? 0

export const assess = (machine: Machine, model: Model, quant: Quant): Assessment => {
  const size = quant.sizeGB
  const label = qualityLabel(model, quant)
  const diskShortGB = Math.max(0, size - machine.freeDiskGB)
  const fast = machine.kind === "unified" ? machine.memoryGB * 0.7 : machine.vramGB - 3
  const host = machine.kind === "unified" ? 0 : Math.max(0, machine.memoryGB - 12)
  const fastName = machine.kind === "unified" ? "Memory" : "GPU"
  const none = { quant, label, placement: { _tag: "DoesNotFit" } as const, speed: 0, tiers: [], diskShortGB, servedK: 0 }

  if (model.activeB === null) {
    const servedK = servedContext(model, size, fast)
    if (servedK === 0) return none
    return { quant, label, placement: { _tag: "Resident", tier: fastName }, speed: 1 / (size / (efficiency.dense * machine.bandwidth) + stepOverhead), tiers: [{ name: fastName, usedGB: size, totalGB: fast }], diskShortGB, servedK }
  }

  // A side table (Qwen3.8's n-grams, DeepSeek's Engram) is read from SSD by design and never held.
  const side = model.sideTable?.gb ?? 0
  const core = size - side
  const always = core * model.denseShare
  const experts = core - always
  const activeBytes = core * model.activeB / (model.paramsB - (model.sideTable?.paramsB ?? 0))
  const activeExperts = activeBytes * (1 - model.denseShare)
  const compute = activeBytes / (efficiency.moe * machine.bandwidth) + stepOverhead
  // The context's memory comes out of the same room the experts would use.
  const servedK = servedContext(model, always + experts * 0.04, fast)
  if (servedK === 0) return none
  const room = fast - servedK * model.kvPerK

  const fastShare = Math.min(1, (room - always) / experts)
  if (fastShare >= 1) return { quant, label, placement: { _tag: "Resident", tier: fastName }, speed: 1 / compute, tiers: [{ name: fastName, usedGB: core, totalGB: fast }, ...(side > 0 ? [{ name: "SSD", usedGB: side, totalGB: null }] : [])], diskShortGB, servedK }

  const hostShare = Math.min(1 - fastShare, host / experts)
  const ssdBytes = missCurve(fastShare + hostShare) * activeExperts
  const ramBytes = Math.max(0, missCurve(fastShare) * activeExperts - ssdBytes)
  const seconds = compute + (machine.pcie > 0 ? ramBytes / machine.pcie : 0) + ssdBytes / machine.ssd
  const inFast = always + experts * fastShare
  const inHost = experts * hostShare
  const onSsd = Math.max(0, core - inFast - inHost) + side
  const tiers = [
    { name: fastName, usedGB: inFast, totalGB: fast },
    ...(inHost > 0.5 ? [{ name: "RAM", usedGB: inHost, totalGB: host }] : []),
    ...(onSsd > 0.5 ? [{ name: "SSD", usedGB: onSsd, totalGB: null }] : []),
  ]
  return { quant, label, placement: { _tag: "Streamed", from: onSsd - side > 0.5 ? "SSD" : "RAM" }, speed: 1 / seconds, tiers, diskShortGB, servedK }
}

/** The quant chosen by default: the highest rung reaching the target speed, else the fastest rung. */
export const chooseQuant = (assessments: readonly Assessment[]): Assessment | undefined => {
  const usable = assessments.filter(a => a.placement._tag !== "DoesNotFit" && a.diskShortGB === 0)
  const fastEnough = usable.filter(a => a.speed >= TARGET_SPEED)
  if (fastEnough.length > 0) return fastEnough[fastEnough.length - 1]
  return [...usable].sort((a, b) => b.speed - a.speed)[0]
}

export interface Ranked {
  readonly model: Model
  readonly assessments: readonly Assessment[]
  readonly chosen: Assessment | undefined
}

export const rankForMachine = (machine: Machine, countries: "Any" | "US"): readonly Ranked[] =>
  pool
    .filter(model => countries === "Any" || model.country === countries)
    .map(model => {
      const assessments = model.quants.map(quant => assess(machine, model, quant))
      return { model, assessments, chosen: chooseQuant(assessments) }
    })
    .sort((a, b) => order(b) - order(a))

const order = ({ model, chosen }: Ranked) =>
  model.comingSoon ? -2 : chosen === undefined ? -1 : (chosen.speed >= TARGET_SPEED ? 1000 : 0) + (model.score ?? 0)

const typicalQuants: readonly (readonly [string, number])[] = [["Q2_K", 3.0], ["Q3_K_M", 3.9], ["Q4_K_M", 4.9], ["Q5_K_M", 5.7], ["Q6_K", 6.6], ["Q8_0", 8.5]]

export interface ListedQuant {
  readonly quant: Quant
  readonly tested: boolean
  readonly runnable: boolean
}

/** Every quant of a model on Hugging Face: the tested ladder first, then each other repo's typical quants. */
export const allQuants = (model: Model, repos: readonly Repo[]): readonly ListedQuant[] => {
  const tested = model.quants.map(quant => ({ quant, tested: true, runnable: true }))
  const others = repos
    .filter(repo => repo.base === model.id && !model.quants.some(quant => quant.repo === repo.repo))
    .flatMap(repo => typicalQuants.map(([name, bits]) => ({
      quant: { publisher: repo.repo.split("/")[0]!, repo: repo.repo, name, sizeGB: Math.round(model.paramsB * bits / 8 * 10) / 10 },
      tested: false,
      runnable: repo.note === undefined,
    })))
  return [...tested, ...others]
}

export const formatContext = (k: number) => k >= 1024 ? `${k / 1024}M` : `${k}K`

export const formatSpeed = (speed: number) => speed < 1 ? "<1 tok/s" : `${Math.round(speed)} tok/s`
export const formatSize = (gb: number) => gb >= 100 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`

export const placementText = (placement: Placement) =>
  placement._tag === "Resident" ? (placement.tier === "GPU" ? "Fits in GPU" : "Fits in memory")
    : placement._tag === "Streamed" ? `Streams from ${placement.from}`
    : "Doesn't fit"
