import type { ProviderModel, ModelProfile } from '@magnitudedev/ai'
import { isProviderModelAvailable, toModelProfile } from '@magnitudedev/ai'
import type { SlotId } from './types'

/**
 * User-facing per-slot config shape (subset that affects model resolution).
 */
export interface UserSlotConfig {
  readonly providerId?: string
  readonly providerModelId?: string
  readonly reasoningEffort?: string
}

/**
 * Result of resolving a slot to a concrete model.
 */
export interface ResolvedSlotModel {
  readonly providerId: string
  readonly providerModelId: string
  readonly profile: ModelProfile
  readonly isUserOverride: boolean
  /** True when the user's override was not found in the catalog. */
  readonly isFallback: boolean
}

/**
 * Resolve a slot to a concrete model ID using the catalog.
 *
 * Works on any `ProviderModel`. The optional `slots` field is
 * application-level routing metadata (only some providers populate it).
 * When no model has `slots` matching the slot, the first model is used.
 *
 * Resolution order:
 * 1. If user has an override modelId AND that model exists in the catalog → use it
 * 2. If user has an override modelId but it's NOT in the catalog → fall back to
 *    the default model for the slot, set isFallback = true.
 * 3. No override → use the default model for the slot from catalog
 * 4. No catalog or no model for the slot → return null
 */
export function resolveSlotModel<T extends ProviderModel & { readonly slots?: readonly SlotId[] }>(
  catalogModels: readonly T[] | null,
  userSlotConfig: UserSlotConfig | undefined,
  slotId: SlotId,
): ResolvedSlotModel | null {
  const overrideProviderId = userSlotConfig?.providerId
  const overrideProviderModelId = userSlotConfig?.providerModelId
  const hasOverride = overrideProviderId !== undefined && overrideProviderModelId !== undefined
  const availableModels = catalogModels?.filter(isProviderModelAvailable) ?? null
  const defaultEntry = availableModels?.find(m => m.slots?.includes(slotId)) ?? availableModels?.[0] ?? null

  if (hasOverride) {
    const overrideEntry = availableModels?.find(
      m => m.providerId === overrideProviderId && m.providerModelId === overrideProviderModelId,
    ) ?? null
    if (overrideEntry) {
      return {
        providerId: overrideEntry.providerId,
        providerModelId: overrideEntry.providerModelId,
        profile: toModelProfile(overrideEntry),
        isUserOverride: true,
        isFallback: false,
      }
    }
    if (!defaultEntry) return null
    return {
      providerId: defaultEntry.providerId,
      providerModelId: defaultEntry.providerModelId,
      profile: toModelProfile(defaultEntry),
      isUserOverride: true,
      isFallback: true,
    }
  }

  if (!defaultEntry) return null
  return {
    providerId: defaultEntry.providerId,
    providerModelId: defaultEntry.providerModelId,
    profile: toModelProfile(defaultEntry),
    isUserOverride: false,
    isFallback: false,
  }
}
