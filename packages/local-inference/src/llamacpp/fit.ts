import { createHash } from "node:crypto"
import { Data, Option, Schema } from "effect"
import type { ModelFileVersionPart } from "../model-files"
import type { LlamaBinary } from "./binary"
import type { LlamaExecutionProfileId, LlamaFitPlanId } from "./identity"
import { LlamaDeviceId, LlamaFitPlanId as FitPlanId } from "./identity"

const MEBIBYTE = 1024 * 1024
export const FIT_DIAGNOSTIC_LIMIT = 16 * 1024

export interface LlamaDevicePlacement {
  readonly device: LlamaDeviceId
  readonly modelBytes: number
  readonly contextBytes: number
  readonly computeBytes: number
}

export interface LlamaFitPlan {
  readonly id: LlamaFitPlanId
  readonly binaryFingerprint: LlamaBinary["fingerprint"]
  readonly profileId: LlamaExecutionProfileId
  readonly fileVersion: readonly ModelFileVersionPart[]
  readonly arguments: readonly string[]
  readonly placement: readonly LlamaDevicePlacement[]
  readonly rawOutput: string
}

export type LlamaFitResult = Data.TaggedEnum<{
  Measured: { readonly plan: LlamaFitPlan }
  Unsupported: { readonly binary: LlamaBinary["fingerprint"] }
  InvalidOutput: { readonly diagnostic: string }
}>
export const LlamaFitResult = Data.taggedEnum<LlamaFitResult>()

const placementLine = /^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$/

export const parseLlamaFitPlacement = (output: string): Option.Option<readonly LlamaDevicePlacement[]> => {
  const placements: LlamaDevicePlacement[] = []
  const devices = new Set<string>()
  for (const line of output.split(/\r?\n/)) {
    const match = placementLine.exec(line.trim())
    if (match === null) continue
    const device = Option.flatMap(Option.fromNullable(match[1]), Schema.decodeUnknownOption(LlamaDeviceId))
    const modelMiB = Number(match[2])
    const contextMiB = Number(match[3])
    const computeMiB = Number(match[4])
    if (Option.isNone(device) || devices.has(device.value) || ![modelMiB, contextMiB, computeMiB].every(Number.isSafeInteger)) return Option.none()
    devices.add(device.value)
    placements.push({
      device: device.value,
      modelBytes: modelMiB * MEBIBYTE,
      contextBytes: contextMiB * MEBIBYTE,
      computeBytes: computeMiB * MEBIBYTE,
    })
  }
  const hostCount = placements.filter(({ device }) => device === "Host").length
  return placements.length > 0 && hostCount === 1 ? Option.some(placements) : Option.none()
}

const canonicalFileVersion = (version: readonly ModelFileVersionPart[]): string => version
  .map((part) => `${part.key}:${part.sizeBytes}:${Option.getOrElse(part.modifiedAtMillis, () => "absent")}`)
  .join("\0")

export const makeLlamaFitPlan = (input: Omit<LlamaFitPlan, "id">): LlamaFitPlan => ({
  ...input,
  id: FitPlanId.make(createHash("sha256").update([
    input.binaryFingerprint,
    input.profileId,
    canonicalFileVersion(input.fileVersion),
    input.arguments.join("\0"),
    input.rawOutput,
  ].join("\0")).digest("hex")),
})

export const boundedFitDiagnostic = (output: string): string => output.slice(0, FIT_DIAGNOSTIC_LIMIT)
