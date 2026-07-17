import { createHash } from "node:crypto"
import { Data, Option, Schema } from "effect"
import type { ModelFileVersionPart } from "../model-files"
import type { LlamaCppInstallation } from "./installation"
import type { LlamaExecutionProfileId, LlamaFitPlanId } from "./identity"
import { LlamaDeviceId, LlamaFitAssessmentKey, LlamaFitPlanId as FitPlanId, NormalizedLlamaModelPath } from "./identity"

const MEBIBYTE = 1024 * 1024
const GIBIBYTE = 1024 * 1024 * 1024
export const FIT_DIAGNOSTIC_LIMIT = 16 * 1024
export const VISION_PROJECTOR_MEMORY_MULTIPLIER = 1.2
export const VISION_FIT_UNCERTAINTY_BYTES = 1.5 * GIBIBYTE
export const LLAMA_FIT_ESTIMATION_POLICY_FINGERPRINT = "llama-fit-additive-projector-2026-07-16"

const NonNegativeFinite = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())
export const LlamaFitDomainAssessmentSchema = Schema.Struct({
  memoryDomainId: Schema.String.pipe(Schema.minLength(1)),
  estimatedBytes: NonNegativeFinite,
  stableCapacityBytes: NonNegativeFinite,
  marginBytes: Schema.Number.pipe(Schema.finite()),
})
export type LlamaFitDomainAssessment = Schema.Schema.Type<typeof LlamaFitDomainAssessmentSchema>

export const LlamaFitAssessmentSchema = Schema.Struct({
  estimatedTotalBytes: NonNegativeFinite,
  domains: Schema.Array(LlamaFitDomainAssessmentSchema).pipe(Schema.minItems(1)),
  result: Schema.Literal("likely_fits", "capacity_risk"),
})
export type LlamaFitAssessment = Schema.Schema.Type<typeof LlamaFitAssessmentSchema>

export const LlamaFitAssessmentCacheEntrySchema = Schema.Struct({
  modelPath: NormalizedLlamaModelPath,
  key: LlamaFitAssessmentKey,
  assessment: LlamaFitAssessmentSchema,
})
export type LlamaFitAssessmentCacheEntry = Schema.Schema.Type<typeof LlamaFitAssessmentCacheEntrySchema>

export interface LlamaDevicePlacement {
  readonly device: LlamaDeviceId
  readonly modelBytes: number
  readonly contextBytes: number
  readonly computeBytes: number
}

export interface LlamaFitPlan {
  readonly id: LlamaFitPlanId
  readonly fitExecutableFingerprint: LlamaCppInstallation["executables"]["fitParams"]["fingerprint"]
  readonly profileId: LlamaExecutionProfileId
  readonly fileVersion: readonly ModelFileVersionPart[]
  readonly arguments: readonly string[]
  readonly placement: readonly LlamaDevicePlacement[]
  readonly memory: LlamaFitMemoryEstimate
  readonly rawOutput: string
}

export interface LlamaVisionFitAdjustment {
  readonly projectorFileBytes: number
  readonly estimatedProjectorBytes: number
  readonly uncertaintyBytes: number
}

export interface LlamaFitMemoryEstimate {
  readonly baseBytes: number
  readonly vision: Option.Option<LlamaVisionFitAdjustment>
  readonly estimatedTotalBytes: number
}

export type LlamaFitResult = Data.TaggedEnum<{
  Estimated: { readonly plan: LlamaFitPlan }
  Unsupported: { readonly fitExecutable: LlamaCppInstallation["executables"]["fitParams"]["fingerprint"] }
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

export const estimateLlamaFitMemory = (
  placement: readonly LlamaDevicePlacement[],
  projectorFileBytes: Option.Option<number>,
): LlamaFitMemoryEstimate => {
  const baseBytes = placement.reduce(
    (total, item) => total + item.modelBytes + item.contextBytes + item.computeBytes,
    0,
  )
  const vision = Option.map(projectorFileBytes, (sizeBytes) => ({
    projectorFileBytes: sizeBytes,
    estimatedProjectorBytes: Math.ceil(sizeBytes * VISION_PROJECTOR_MEMORY_MULTIPLIER),
    uncertaintyBytes: VISION_FIT_UNCERTAINTY_BYTES,
  }))
  return {
    baseBytes,
    vision,
    estimatedTotalBytes: baseBytes + Option.match(vision, {
      onNone: () => 0,
      onSome: ({ estimatedProjectorBytes, uncertaintyBytes }) => estimatedProjectorBytes + uncertaintyBytes,
    }),
  }
}

const canonicalFileVersion = (version: readonly ModelFileVersionPart[]): string => version
  .map((part) => `${part.key}:${part.sizeBytes}:${Option.getOrElse(part.modifiedAtMillis, () => "absent")}`)
  .join("\0")

export const makeLlamaFitAssessmentKey = (input: {
  readonly modelPath: NormalizedLlamaModelPath
  readonly fileVersion: readonly ModelFileVersionPart[]
  readonly projectorPath: Option.Option<string>
  readonly profileId: LlamaExecutionProfileId
  readonly fitExecutableFingerprint: LlamaCppInstallation["executables"]["fitParams"]["fingerprint"]
  readonly hardwareFingerprint: string
}): LlamaFitAssessmentKey => LlamaFitAssessmentKey.make(createHash("sha256").update([
  input.modelPath,
  canonicalFileVersion(input.fileVersion),
  Option.getOrElse(input.projectorPath, () => "<absent>"),
  input.profileId,
  input.fitExecutableFingerprint,
  input.hardwareFingerprint,
  LLAMA_FIT_ESTIMATION_POLICY_FINGERPRINT,
].join("\0")).digest("hex"))

export const makeLlamaFitPlan = (input: Omit<LlamaFitPlan, "id">): LlamaFitPlan => ({
  ...input,
  id: FitPlanId.make(createHash("sha256").update([
    input.fitExecutableFingerprint,
    input.profileId,
    canonicalFileVersion(input.fileVersion),
    input.arguments.join("\0"),
    String(input.memory.estimatedTotalBytes),
    input.rawOutput,
  ].join("\0")).digest("hex")),
})

export const boundedFitDiagnostic = (output: string): string => output.slice(0, FIT_DIAGNOSTIC_LIMIT)
