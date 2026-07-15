export const NO_PROVIDERS_CONFIGURED_MESSAGE = "No providers configured"

interface ModelSlotSelection {
  readonly providerId?: string
  readonly providerModelId?: string
}

export function hasExplicitModelSlots(slotConfig: {
  readonly primary?: ModelSlotSelection
  readonly secondary?: ModelSlotSelection
} | null): boolean {
  return Boolean(
    slotConfig?.primary?.providerId
    && slotConfig.primary.providerModelId
    && slotConfig.secondary?.providerId
    && slotConfig.secondary.providerModelId,
  )
}

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
