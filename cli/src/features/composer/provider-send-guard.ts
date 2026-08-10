export const NO_MODEL_SELECTED_MESSAGE = "Select a model before sending"

/**
 * Returns whether a message may be submitted. The caller must run this before
 * clearing the draft so a rejected submission leaves the user's text intact.
 */
export function allowModelMessageSend(
  modelsConfigured: boolean,
  showToast: (message: string) => void,
): boolean {
  if (modelsConfigured) return true
  showToast(NO_MODEL_SELECTED_MESSAGE)
  return false
}
