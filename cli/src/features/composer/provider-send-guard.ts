export const NO_PROVIDERS_CONFIGURED_MESSAGE = "No providers configured"

/**
 * Returns whether a message may be submitted. The caller must run this before
 * clearing the draft so a rejected submission leaves the user's text intact.
 */
export function allowProviderMessageSend(
  modelsConfigured: boolean,
  showToast: (message: string) => void,
): boolean {
  if (modelsConfigured) return true
  showToast(NO_PROVIDERS_CONFIGURED_MESSAGE)
  return false
}
