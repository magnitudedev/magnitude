import { Schema } from "effect"

const id = <const Brand extends string>(brand: Brand, max = 1024) => Schema.String.pipe(Schema.minLength(1), Schema.maxLength(max), Schema.brand(brand))
export const LlamaDistributionVariantId = id("LlamaDistributionVariantId")
export type LlamaDistributionVariantId = Schema.Schema.Type<typeof LlamaDistributionVariantId>
export const LlamaCppExecutableFingerprint = id("LlamaCppExecutableFingerprint", 256)
export type LlamaCppExecutableFingerprint = Schema.Schema.Type<typeof LlamaCppExecutableFingerprint>
export const LlamaCppInstallationId = id("LlamaCppInstallationId", 256)
export type LlamaCppInstallationId = Schema.Schema.Type<typeof LlamaCppInstallationId>
export const LlamaInstallOperationId = id("LlamaInstallOperationId", 256)
export type LlamaInstallOperationId = Schema.Schema.Type<typeof LlamaInstallOperationId>
export const LlamaBuildCommitId = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{7,64}$/i), Schema.brand("LlamaBuildCommitId"))
export type LlamaBuildCommitId = Schema.Schema.Type<typeof LlamaBuildCommitId>
export const LlamaDeviceId = id("LlamaDeviceId", 512)
export type LlamaDeviceId = Schema.Schema.Type<typeof LlamaDeviceId>
export const LlamaInstanceId = id("LlamaInstanceId")
export type LlamaInstanceId = Schema.Schema.Type<typeof LlamaInstanceId>
export const LlamaServedModelId = id("LlamaServedModelId", 4096)
export type LlamaServedModelId = Schema.Schema.Type<typeof LlamaServedModelId>
export const LlamaModelRegistrationId = id("LlamaModelRegistrationId")
export type LlamaModelRegistrationId = Schema.Schema.Type<typeof LlamaModelRegistrationId>
export const LlamaExecutionProfileId = id("LlamaExecutionProfileId")
export type LlamaExecutionProfileId = Schema.Schema.Type<typeof LlamaExecutionProfileId>
export const LlamaFitPlanId = id("LlamaFitPlanId")
export type LlamaFitPlanId = Schema.Schema.Type<typeof LlamaFitPlanId>
export const LlamaFitAssessmentKey = id("LlamaFitAssessmentKey")
export type LlamaFitAssessmentKey = Schema.Schema.Type<typeof LlamaFitAssessmentKey>
export const LlamaOperationId = id("LlamaOperationId")
export type LlamaOperationId = Schema.Schema.Type<typeof LlamaOperationId>
export const ExternalServerConfigId = id("ExternalServerConfigId")
export type ExternalServerConfigId = Schema.Schema.Type<typeof ExternalServerConfigId>

export const NormalizedLlamaModelPath = Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(16_384),
  Schema.filter((value) => value !== "none" && !value.includes("\0")),
  Schema.brand("NormalizedLlamaModelPath"),
)
export type NormalizedLlamaModelPath = Schema.Schema.Type<typeof NormalizedLlamaModelPath>

/**
 * Lexical, host-independent llama.cpp model-path normalization. This must be
 * used for both managed primary paths and externally reported model_path.
 */
export const normalizeLlamaModelPath = (input: string): NormalizedLlamaModelPath | undefined => {
  const trimmed = input.trim()
  if (trimmed.length === 0 || trimmed === "none" || trimmed.includes("\0")) return undefined
  const slash = trimmed.replaceAll("\\", "/").normalize("NFC")
  const unc = slash.startsWith("//")
  const drive = /^([A-Za-z]):(\/|$)/.exec(slash)
  const absolute = slash.startsWith("/") || drive !== null
  const prefix = unc ? "//" : drive ? `${drive[1]!.toUpperCase()}:` : absolute ? "/" : ""
  const body = drive ? slash.slice(2) : slash.replace(/^\/+/, "")
  const segments: string[] = []
  for (const segment of body.split("/")) {
    if (segment === "" || segment === ".") continue
    if (segment === "..") {
      if (segments.length > 0 && segments.at(-1) !== "..") segments.pop()
      else if (!absolute) segments.push(segment)
      continue
    }
    segments.push(segment)
  }
  const separator = drive !== null && segments.length > 0 ? "/" : ""
  const normalized = drive !== null && segments.length === 0
    ? `${prefix}/`
    : `${prefix}${separator}${segments.join("/")}` || (absolute ? prefix : ".")
  return Schema.decodeUnknownOption(NormalizedLlamaModelPath)(normalized).pipe(
    (option) => option._tag === "Some" ? option.value : undefined,
  )
}

export const isAbsoluteLlamaModelPath = (path: NormalizedLlamaModelPath): boolean =>
  path.startsWith("/") || path.startsWith("//") || /^[A-Z]:\//.test(path)
