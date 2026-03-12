import { Effect, Layer } from 'effect'
import { detectProviders } from '../detect'
import { getModelContextWindow, peekSlot, getSlots } from '../state/provider-state'
import type { ModelSlot } from '../state/provider-state'
import { Model } from '../model/model'
import type { InferenceConfig } from '../model/inference-config'
import { BamlDriver, ResponsesDriver } from '../drivers'
import { ensureAuth } from './ensure-auth'

import { createBoundModel } from './pipeline'
import { ModelResolver } from './model-resolver'
import { NotConfigured, ProviderDisconnected } from '../errors/model-error'
import { getProvider } from '../registry'


const COMPACT_TRIGGER_RATIO = 0.9

function toModel(slot: ModelSlot) {
  return peekSlot(slot)?.model ?? null
}

export function makeModelResolver(): Layer.Layer<ModelResolver> {
  return Layer.effect(
    ModelResolver,
    Effect.gen(function* () {
      return {
        resolve: (slot: ModelSlot) =>
          Effect.gen(function* () {
            const initialModel = toModel(slot)
            if (!initialModel) {
              return yield* Effect.fail(new NotConfigured({ message: `No model configured for slot: ${slot}` }))
            }

            const connected = new Set(detectProviders().map((p) => p.provider.id))
            if (!connected.has(initialModel.providerId)) {
              const providerName = getProvider(initialModel.providerId)?.name ?? initialModel.providerId
              return yield* Effect.fail(new ProviderDisconnected({
                providerId: initialModel.providerId,
                providerName,
                message: `${providerName} is not connected. Please connect the provider or choose another provider/model in /settings.`,
              }))
            }

            yield* ensureAuth(slot)

            const model = toModel(slot)
            if (!model) {
              return yield* Effect.fail(new NotConfigured({ message: `No model configured for slot: ${slot}` }))
            }

            const s = getSlots()[slot]
            const auth = s.auth
            const isCodex = model.providerId === 'openai' && auth?.type === 'oauth'
            const isCopilotCodex = model.providerId === 'github-copilot' && model.id.includes('codex')
            const driver = (isCodex || isCopilotCodex) ? ResponsesDriver : BamlDriver
            const inference: InferenceConfig = {}
            const connection = yield* driver.connect(model, auth, inference)
            return createBoundModel(slot, model, connection, driver, inference)
          }),
        peek: (slot: ModelSlot = 'primary') => toModel(slot),
        contextLimits: (slot: ModelSlot = 'primary') => {
          const hardCap = getModelContextWindow(slot)
          return {
            hardCap,
            softCap: Math.floor(hardCap * COMPACT_TRIGGER_RATIO),
          }
        },
        contextWindow: (slot: ModelSlot = 'primary') => getModelContextWindow(slot),
      }
    }),
  )
}