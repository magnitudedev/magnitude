import {
  deriveOnboardingModelSetupView,
  type OnboardingModelSetupView,
} from "@magnitudedev/client-common"
export { deriveOnboardingModelSetupView }

export const deriveModelSetupActive = ({
  forceSetup,
  onboardingRequired,
  completionSucceeded,
}: {
  readonly forceSetup: boolean
  readonly onboardingRequired: boolean
  readonly completionSucceeded: boolean
}): boolean => {
  if (onboardingRequired) return true
  return forceSetup && !completionSucceeded
}

export const onboardingModelSetupPlaceholder = (view: OnboardingModelSetupView): string | null => {
  switch (view._tag) {
    case "Inactive": return null
    case "Choosing": return "Select a model to start coding…"
    case "Downloading": return `Downloading ${view.candidate.displayName}…`
    case "DownloadFailed": return `Couldn’t download ${view.candidate.displayName}`
    case "Configuring": return `Configuring ${view.candidate.displayName}…`
    case "Activating":
      return view.phase === "Loading"
        ? `Loading ${view.displayName}…`
        : view.phase === "Stopping"
          ? `Stopping ${view.displayName}…`
          : view.phase === "Ready"
            ? `Finishing setup for ${view.displayName}…`
            : `Couldn’t load ${view.displayName}`
  }
}
