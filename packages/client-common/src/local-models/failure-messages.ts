import type { DrafterNotUsable, ModelAcquisitionFailure, ModelLocationKind, ModelVersionNotRunnable, UnsupportedReason } from "@magnitudedev/sdk"
import { formatMemorySize, formatStorageSize } from "../utils/format-bytes"
import { huggingFaceAccessMessage } from "../hugging-face/account"

export const modelDownloadFailureMessage = (failure: ModelAcquisitionFailure): string => {
  switch (failure._tag) {
    case "Interrupted": return "The download was interrupted. Try again to continue."
    case "InsufficientDiskSpace":
      return `Not enough disk space. Free at least ${formatStorageSize(
        Math.max(0, failure.requiredBytes - failure.availableBytes),
      )} and try again.`
    case "SourceUnavailable": return "This model is not available from its source."
    case "HuggingFaceAccessRequired": return huggingFaceAccessMessage(failure.access)
    case "NetworkUnavailable":
      return "Couldn’t reach the model source. Check your connection and try again."
    case "CorruptDownload": return "The downloaded file couldn’t be verified. Try the download again."
    case "LocalStorageFailure":
      return "Magnitude couldn’t write the model to disk. Check disk access and try again."
    case "Internal": return failure.message
  }
}

/** What users read for one reason the engine can't run a model; `detail` stays developer text. */
export const unsupportedReasonMessage = (reason: UnsupportedReason): string => {
  switch (reason._tag) {
    case "Architecture": return `No kernels for the ${reason.architecture} architecture`
    case "WeightTypes":
      return `Uses ${joinWithAnd(reason.types.map(({ ggmlType }) => ggmlType))} weights, which Magnitude doesn't support yet`
    case "RopeScaling": return "Uses rope frequency scaling, which Magnitude doesn't support yet"
    case "UnrecognizedTensor": return `Has parts Magnitude doesn't recognize (${reason.name})`
    case "UnrecognizedMetadata": return `Has parts Magnitude doesn't recognize (${reason.key})`
    case "MissingMetadata": return `Is missing a setting Magnitude needs (${reason.key})`
    case "Tokenizer": return "Uses a tokenizer Magnitude doesn't support"
    case "ChatTemplate": return "Its chat template can't be read"
    case "TiedOutputHead": return "Uses a tied output layer, which Magnitude doesn't support yet"
    case "TargetMismatch": return "Its drafter wasn't built for this model"
    case "KernelDomain":
    case "Backend": return `Its layer sizes aren't supported on this ${backendDevice(reason.backend)}`
    case "Unreadable": return "Magnitude can't read this file"
    case "UnsupportedFormat": return "Stored in a format Magnitude can't read yet"
    case "Unknown": return reason.detail.trim() === "" ? "Magnitude can't run this model" : `Magnitude can't run this model (${reason.detail.trim()})`
  }
}

/** The device a backend reason names: the CPU, a GPU, or the computer when the engine names no backend. */
const backendDevice = (backend: string): string => {
  switch (backend.toLowerCase()) {
    case "cpu": return "CPU"
    case "metal":
    case "cuda":
    case "vulkan": return "GPU"
    default: return "computer"
  }
}

/**
 * Identical reasons collapsed to one; unrecognized parts collapse by kind, counting each distinct part once however
 * many verdicts name it.
 */
export interface UnsupportedReasonGroup {
  readonly reason: UnsupportedReason
  readonly count: number
}

const unrecognizedPart = (reason: UnsupportedReason): string | undefined =>
  reason._tag === "UnrecognizedTensor" ? reason.name : reason._tag === "UnrecognizedMetadata" ? reason.key : undefined

/** The engine's reasons collapsed into groups, in the order each first appears. */
export const unsupportedReasonGroups = (reasons: ReadonlyArray<UnsupportedReason>): ReadonlyArray<UnsupportedReasonGroup> => {
  const groups = new Map<string, { readonly reason: UnsupportedReason; readonly parts: Set<string> }>()
  for (const reason of reasons) {
    const part = unrecognizedPart(reason)
    const key = part === undefined ? unsupportedReasonMessage(reason) : reason._tag
    const group = groups.get(key) ?? { reason, parts: new Set<string>() }
    group.parts.add(part ?? key)
    groups.set(key, group)
  }
  return [...groups.values()].map(({ reason, parts }) => ({ reason, count: parts.size }))
}

/** One group's reason: "Has parts Magnitude doesn't recognize (60 tensors)" for many unrecognized parts. */
export const unsupportedReasonGroupMessage = ({ reason, count }: UnsupportedReasonGroup): string => {
  if (count > 1 && reason._tag === "UnrecognizedTensor") return `Has parts Magnitude doesn't recognize (${count} tensors)`
  if (count > 1 && reason._tag === "UnrecognizedMetadata") return `Has parts Magnitude doesn't recognize (${count} metadata keys)`
  return unsupportedReasonMessage(reason)
}

/** Every distinct user-facing reason, in the engine's order, with identical reasons said once. */
export const unsupportedReasonMessages = (reasons: ReadonlyArray<UnsupportedReason>): ReadonlyArray<string> =>
  unsupportedReasonGroups(reasons).map(unsupportedReasonGroupMessage)

/** Why a Hugging Face search result can't be added: the architecture Magnitude doesn't run yet, or model files it can't read. */
export const searchResultUnsupportedMessage = (reasons: ReadonlyArray<UnsupportedReason>): string => {
  const architecture = reasons.find((reason) => reason._tag === "Architecture")
  if (architecture?._tag === "Architecture") return `Magnitude doesn't run ${architecture.architecture} yet`
  if (reasons.some((reason) => reason._tag === "Unreadable")) return "Magnitude can't read this repository's model files"
  return unsupportedReasonMessages(reasons).join(". ")
}

/** The label on a search result whose files need access granted on Hugging Face. */
export const GATED_SEARCH_RESULT_LABEL = "Requires access on Hugging Face"

/** Why a drafter was refused for a model, in the client's words for the engine's reasons. */
export const drafterNotUsableMessage = ({ reasons }: Pick<DrafterNotUsable, "reasons">): string =>
  reasons.length === 0
    ? "This drafter can't be used with this model."
    : `This drafter can't be used with this model: ${unsupportedReasonMessages(reasons).join("; ")}.`

/** Why a version was refused before anything downloaded. */
export const modelVersionNotRunnableMessage = ({ version, verdict }: Pick<ModelVersionNotRunnable, "version" | "verdict">): string =>
  verdict._tag === "Unsupported"
    ? `Magnitude can't run ${version}: ${unsupportedReasonMessages(verdict.reasons).join("; ")}.`
    : `${version} needs ${formatMemorySize(verdict.deficitBytes, { rounding: "up" })} more memory than this computer can provide.`

const joinWithAnd = (values: ReadonlyArray<string>): string =>
  values.length <= 2 ? values.join(" and ") : `${values.slice(0, -1).join(", ")}, and ${values.at(-1)}`

/** Why a model location reads "Can’t read", and what the user can do about it. */
export const unreadableModelLocationMessage = (kind: ModelLocationKind): string =>
  kind === "Ollama"
    ? "Magnitude doesn’t have permission to read this folder. A system Ollama service keeps its models in a folder that belongs to the ollama user: give your account read access to it, or set OLLAMA_MODELS to a folder you can read (Magnitude also scans the default folders)."
    : "Magnitude doesn’t have permission to read this folder. Give your account read access to it."
