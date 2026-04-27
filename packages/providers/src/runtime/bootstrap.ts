import { Effect } from 'effect'
import { AppConfig } from '@magnitudedev/storage'
import { ProviderAuth, ProviderCatalog, ProviderState } from './contracts'

export function bootstrapProviderRuntime<TSlot extends string>(args: {
  slots: readonly TSlot[]
  validateSelection?: (slot: TSlot, providerId: string, modelId: string) => boolean
}): Effect.Effect<void, never, ProviderCatalog | ProviderAuth | AppConfig | ProviderState> {
  return Effect.gen(function* () {
    const catalog = yield* ProviderCatalog
    const config = yield* AppConfig
    const auth = yield* ProviderAuth
    const state = yield* ProviderState

    yield* catalog.refresh()

    const connectedIds = yield* auth.connectedProviderIds()

    for (const slot of args.slots) {
      const selection = yield* config.getModelSelection(slot)
      if (!selection) continue

      const modelDef = yield* catalog.getModel(selection.providerId, selection.modelId)
      const isConnected = connectedIds.has(selection.providerId)
      const isActive = Boolean(modelDef)
      const isValidForSlot = args.validateSelection?.(slot, selection.providerId, selection.modelId) ?? true

      if (!isConnected || !isActive || !isValidForSlot) {
        yield* config.setModelSelection(slot, null)
        continue
      }

      const providerAuth = (yield* auth.getAuth(selection.providerId)) ?? null
      yield* state.setSelection(slot, selection.providerId, selection.modelId, providerAuth, { persist: false })
    }
  })
}