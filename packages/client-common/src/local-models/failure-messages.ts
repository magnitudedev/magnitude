import type { ModelAcquisitionFailure } from "@magnitudedev/sdk"
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
