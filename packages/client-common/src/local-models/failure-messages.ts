import type { ModelAcquisitionFailure, UnsupportedReason } from "@magnitudedev/sdk"
import { formatStorageSize } from "../utils/format-bytes"

export const modelDownloadFailureMessage = (failure: ModelAcquisitionFailure): string => {
  switch (failure._tag) {
    case "Interrupted": return "The download was interrupted. Try again to continue."
    case "InsufficientDiskSpace":
      return `Not enough disk space. Free at least ${formatStorageSize(
        Math.max(0, failure.requiredBytes - failure.availableBytes),
      )} and try again.`
    case "SourceUnavailable": return "This model is not available from its source."
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
    case "Tokenizer": return "Uses a tokenizer Magnitude doesn't support"
    case "ChatTemplate": return "Its chat template can't be read"
    case "TiedOutputHead": return "Uses a tied output layer, which Magnitude doesn't support yet"
    case "TargetMismatch": return "Its drafter wasn't built for this model"
    case "KernelDomain":
    case "Backend": return "Its layer sizes aren't supported on this GPU"
    case "Unreadable": return "The file is incomplete or damaged"
    case "Unknown": return "Magnitude can't run this model"
  }
}

/** Every distinct user-facing reason, in the engine's order. */
export const unsupportedReasonMessages = (reasons: ReadonlyArray<UnsupportedReason>): ReadonlyArray<string> =>
  [...new Set(reasons.map(unsupportedReasonMessage))]

const joinWithAnd = (values: ReadonlyArray<string>): string =>
  values.length <= 2 ? values.join(" and ") : `${values.slice(0, -1).join(", ")}, and ${values.at(-1)}`
