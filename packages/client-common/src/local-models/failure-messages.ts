import type { ModelDownloadFailure } from "@magnitudedev/sdk"
import type {
  OnboardingModelSetupFailure,
  OnboardingModelSetupNotice,
  OnboardingModelSetupExecution,
} from "./setup-state"
import { formatLocalModelDisplayName } from "../utils/model-presentation"
import { formatStorageSize } from "../utils/format-bytes"

export const modelDownloadFailureMessage = (failure: ModelDownloadFailure): string => {
  switch (failure._tag) {
    case "Interrupted": return "The download was interrupted. Try again to continue."
    case "InsufficientDiskSpace":
      return `Not enough disk space. Free at least ${formatStorageSize(
        Math.max(0, failure.requiredBytes - failure.availableBytes),
      )} and try again.`
    case "SourceUnavailable": return "This model is not available from its source."
    case "NetworkUnavailable":
      return "Couldn’t reach the model source. Check your connection and try again."
    case "CorruptDownload":
      return "The downloaded file couldn’t be verified. Try the download again."
    case "LocalStorageFailure":
      return "Magnitude couldn’t write the model to disk. Check disk access and try again."
    case "Internal": return "Magnitude couldn’t complete the download. Try again."
  }
}

const onboardingModelSetupFailureDetail = (
  failure: OnboardingModelSetupFailure,
): string => {
  if (typeof failure !== "object" || failure === null) {
    return "The onboarding model setup could not be completed."
  }
  if (!("_tag" in failure)) {
    const message = "message" in failure ? failure.message : undefined
    return typeof message === "string" ? message : "The onboarding model setup could not be completed."
  }
  switch (failure._tag) {
    case "OnboardingModelChoiceRejected":
      return "That model is no longer available for setup."
    case "OnboardingModelResourceChanged":
      return "The selected model changed before setup completed. Choose it again to retry."
    case "Interrupted":
    case "InsufficientDiskSpace":
    case "SourceUnavailable":
    case "NetworkUnavailable":
    case "CorruptDownload":
    case "LocalStorageFailure":
    case "Internal": return modelDownloadFailureMessage(failure as ModelDownloadFailure)
    default: {
      const message = "message" in failure ? failure.message : undefined
      return typeof message === "string" && message.length > 0
        ? message
        : "The onboarding model setup could not be completed."
    }
  }
}

const modelOperationVerb = (
  operation: OnboardingModelSetupExecution["_tag"],
): string => {
  switch (operation) {
    case "Preparing": return "preparing"
    case "Installing": return "downloading"
    case "Configuring": return "configuring"
    case "Loading": return "loading"
    case "Completing": return "finishing setup for"
  }
}

export const onboardingModelSetupNoticeMessage = (
  notice: OnboardingModelSetupNotice,
): string => {
  const detail = onboardingModelSetupFailureDetail(notice.failure)
  if (notice.subject._tag === "Setup") return `Unexpected setup error · ${detail}`
  return `Unexpected error ${modelOperationVerb(notice.subject.operation)} ${
    formatLocalModelDisplayName(notice.subject.model)
  } · ${detail}`
}
