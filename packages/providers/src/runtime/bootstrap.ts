import { Effect } from 'effect'
import { AppConfig } from '@magnitudedev/storage'
import { isBrowserCompatible } from '../browser-models'
import { ProviderAuth, ProviderCatalog, ProviderState } from './contracts'

export const bootstrapProviderRuntime = Effect.gen(function* () {
  const catalog = yield* ProviderCatalog
  const config = yield* AppConfig
  const auth = yield* ProviderAuth
  const state = yield* ProviderState

  yield* catalog.refresh()

  const primaryModel = yield* config.getModelSelection('primary')
  const secondaryModel = yield* config.getModelSelection('secondary')
  const browserModel = yield* config.getModelSelection('browser')

  const connectedIds = yield* auth.connectedProviderIds()

  if (primaryModel) {
    if (connectedIds.has(primaryModel.providerId)) {
      const providerAuth = (yield* auth.getAuth(primaryModel.providerId)) ?? null
      yield* state.setSelection(
        'primary',
        primaryModel.providerId,
        primaryModel.modelId,
        providerAuth,
        { persist: false },
      )
    } else {
      yield* config.setModelSelection('primary', null)
    }
  }

  if (secondaryModel) {
    if (connectedIds.has(secondaryModel.providerId)) {
      const providerAuth = (yield* auth.getAuth(secondaryModel.providerId)) ?? null
      yield* state.setSelection(
        'secondary',
        secondaryModel.providerId,
        secondaryModel.modelId,
        providerAuth,
        { persist: false },
      )
    } else {
      yield* config.setModelSelection('secondary', null)
    }
  }

  if (browserModel) {
    const browserDef = yield* catalog.getModel(browserModel.providerId, browserModel.modelId)
    if (
      connectedIds.has(browserModel.providerId) &&
      browserDef &&
      browserDef.status !== 'deprecated' &&
      isBrowserCompatible(browserModel.providerId, browserModel.modelId)
    ) {
      const providerAuth = (yield* auth.getAuth(browserModel.providerId)) ?? null
      yield* state.setSelection(
        'browser',
        browserModel.providerId,
        browserModel.modelId,
        providerAuth,
        { persist: false },
      )
    } else {
      yield* config.setModelSelection('browser', null)
    }
  }
})