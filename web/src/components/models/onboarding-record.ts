const KEY = "magnitude.onboarding.completed"

/**
 * Whether this window last saw onboarding completed, so a first run opens onboarding before the service answers and a
 * later one doesn't flash it. The service's answer always replaces it.
 */
export const onboardingCompletedHere = (): boolean => {
  try { return localStorage.getItem(KEY) === "true" } catch { return false }
}

export const recordOnboardingCompleted = (completed: boolean): void => {
  try {
    if (completed) localStorage.setItem(KEY, "true")
    else localStorage.removeItem(KEY)
  } catch {}
}
