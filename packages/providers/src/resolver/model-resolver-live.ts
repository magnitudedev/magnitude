import { Effect, Layer } from 'effect'
import { AppConfig } from '@magnitudedev/storage'
import type { InferenceConfig } from '../model/inference-config'
import { BamlDriver, ResponsesDriver } from '../drivers'
import { ensureAuth } from './ensure-auth'
import { createBoundModel } from './pipeline'
import { ModelResolver, type ModelResolverShape } from './model-resolver'
import { NotConfigured, ProviderDisconnected } from '../errors/model-error'
import { ProviderAuth, ProviderCatalog, ProviderState } from '../runtime/contracts'

export function makeModelResolver<TSlot extends string>(): Layer.Layer<ModelResolver, never, ProviderCatalog | ProviderState | ProviderAuth | AppConfig> {
  return Layer.effect(
    ModelResolver,
    Effect.gen(function* () {
      const catalog = yield* ProviderCatalog
      const state = yield* ProviderState
      const auth = yield* ProviderAuth
      const appConfig = yield* AppConfig

      const resolver: ModelResolverShape<TSlot> = {
        resolve: (slot) =>
          Effect.gen(function* () {
            const initial = yield* state.peek(slot)
            if (!initial) {
              return yield* Effect.fail(new NotConfigured({ message: `No model configured for slot: ${slot}` }))
            }

            const connected = yield* auth.connectedProviderIds()
            if (!connected.has(initial.model.providerId)) {
              const providerName = yield* catalog.getProviderName(initial.model.providerId)
              return yield* Effect.fail(new ProviderDisconnected({
                providerId: initial.model.providerId,
                providerName,
                message: `${providerName} is not connected. Please connect the provider or choose another provider/model in /settings.`,
              }))
            }

            yield* ensureAuth(slot).pipe(
              Effect.provideService(ProviderState, state),
              Effect.provideService(ProviderAuth, auth),
            )

            const current = yield* state.peek(slot)
            if (!current) {
              return yield* Effect.fail(new NotConfigured({ message: `No model configured for slot: ${slot}` }))
            }

            const { model, auth: currentAuth } = current
            const isCodex = model.providerId === 'openai' && currentAuth?.type === 'oauth'
            const isCopilotCodex = model.providerId === 'github-copilot' && model.id.includes('codex')
            const driver = (isCodex || isCopilotCodex) ? ResponsesDriver : BamlDriver
            const inference: InferenceConfig = {}
            const connection = yield* driver.connect(model, currentAuth, inference)
            const config = yield* appConfig.load()
            return yield* createBoundModel(
              slot,
              model,
              connection,
              driver,
              inference,
              config.providers?.[model.providerId],
            ).pipe(Effect.provideService(ProviderState, state))
          }),
      }

      return resolver as ModelResolverShape<string>
    }),
  )
}