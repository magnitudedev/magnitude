import type { WizardStep } from '../components/setup-wizard-overlay'

export interface WizardModelSelection {
  providerId: string
  modelId: string
}

export interface ResolveLocalWizardSlotDefaultsArgs<TSlot extends string> {
  slots: readonly TSlot[]
  providerId: string
  existingSlotModels: Record<TSlot, WizardModelSelection | null>
  discoveredModelIds?: string[]
  rememberedModelIds?: string[]
  applyWizardDefaults: boolean
}

function normalizeModelId(value: string): string {
  return value.trim()
}

function resolveLocalWizardDefaultModelId<TSlot extends string>({
  providerId,
  slots,
  existingSlotModels,
  discoveredModelIds = [],
  rememberedModelIds = [],
}: Omit<ResolveLocalWizardSlotDefaultsArgs<TSlot>, 'applyWizardDefaults'>): string | null {
  for (const slot of slots) {
    const selection = existingSlotModels[slot]
    if (selection?.providerId === providerId) {
      const id = normalizeModelId(selection.modelId)
      if (id.length > 0) return id
    }
  }

  const discovered = discoveredModelIds.map(normalizeModelId).find((id) => id.length > 0)
  if (discovered) return discovered

  const remembered = rememberedModelIds.map(normalizeModelId).find((id) => id.length > 0)
  if (remembered) return remembered

  return null
}

export function resolveLocalWizardSlotDefaults<TSlot extends string>({
  slots,
  providerId,
  existingSlotModels,
  discoveredModelIds = [],
  rememberedModelIds = [],
  applyWizardDefaults,
}: ResolveLocalWizardSlotDefaultsArgs<TSlot>): Record<TSlot, WizardModelSelection | null> {
  if (!applyWizardDefaults) {
    return { ...existingSlotModels }
  }

  const selectedModelId = resolveLocalWizardDefaultModelId({
    providerId,
    slots,
    existingSlotModels,
    discoveredModelIds,
    rememberedModelIds,
  })

  const next = { ...existingSlotModels }
  if (!selectedModelId) {
    for (const slot of slots) next[slot] = null
    return next
  }

  for (const slot of slots) {
    next[slot] = { providerId, modelId: selectedModelId }
  }
  return next
}

export function computeWizardTotalSteps(
  wizardNeedsChromium: boolean | null,
  wizardHasProviderEndpointStep: boolean,
): number {
  const baseSteps = wizardHasProviderEndpointStep ? 3 : 2
  if (wizardNeedsChromium === false) return baseSteps
  return baseSteps + 1
}

export function resolveWizardBackStep(
  wizardStep: WizardStep,
  wizardHasProviderEndpointStep: boolean,
): WizardStep {
  if (wizardStep === 'browser') return 'models'
  if (wizardStep === 'models' && wizardHasProviderEndpointStep) return 'provider-endpoint'
  if (wizardStep === 'provider-endpoint') return 'provider'
  return 'provider'
}
