import { Effect } from 'effect'
import { isBrowserCompatible } from '../browser-models'
import { ProviderAuth, ProviderCatalog, ProviderConfig, ProviderState } from './contracts'

export const bootstrapProviderRuntime = Effect.gen(function* () {
  const catalog = yield* ProviderCatalog
  const config = yield* ProviderConfig
  const auth = yield* ProviderAuth
  const state = yield* ProviderState

  yield* catalog.refresh()

  const currentConfig = yield* config.loadConfig()
  const local = yield* config.getLocalProviderConfig()

  const localOpts = currentConfig.providerOptions?.['local']
  if ((!local.baseUrl || !local.modelId) && localOpts?.baseUrl && localOpts?.modelId) {
    yield* config.setLocalProviderConfig(localOpts.baseUrl, localOpts.modelId)
  }

  const connectedIds = yield* auth.connectedProviderIds()
  let configChanged = false

  if (currentConfig.primaryModel) {
    if (connectedIds.has(currentConfig.primaryModel.providerId)) {
      const providerAuth = (yield* auth.getAuth(currentConfig.primaryModel.providerId)) ?? null
      yield* state.setSelection(
        'primary',
        currentConfig.primaryModel.providerId,
        currentConfig.primaryModel.modelId,
        providerAuth,
        { persist: false },
      )
    } else {
      currentConfig.primaryModel = null
      configChanged = true
    }
  }

  if (currentConfig.secondaryModel) {
    if (connectedIds.has(currentConfig.secondaryModel.providerId)) {
      const providerAuth = (yield* auth.getAuth(currentConfig.secondaryModel.providerId)) ?? null
      yield* state.setSelection(
        'secondary',
        currentConfig.secondaryModel.providerId,
        currentConfig.secondaryModel.modelId,
        providerAuth,
        { persist: false },
      )
    } else {
      currentConfig.secondaryModel = null
      configChanged = true
    }
  }

  if (currentConfig.browserModel) {
    if (
      connectedIds.has(currentConfig.browserModel.providerId) &&
      isBrowserCompatible(currentConfig.browserModel.providerId, currentConfig.browserModel.modelId)
    ) {
      const providerAuth = (yield* auth.getAuth(currentConfig.browserModel.providerId)) ?? null
      yield* state.setSelection(
        'browser',
        currentConfig.browserModel.providerId,
        currentConfig.browserModel.modelId,
        providerAuth,
        { persist: false },
      )
    } else {
      currentConfig.browserModel = null
      configChanged = true
    }
  }

  if (configChanged) {
    yield* config.saveConfig(currentConfig)
  }
})