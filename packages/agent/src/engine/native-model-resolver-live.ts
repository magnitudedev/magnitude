/**
 * NativeModelResolverLive — real implementation of NativeModelResolver.
 *
 * Reads ProviderState for the slot's model + stored auth, falls back to env
 * for API-key providers, then constructs a NativeBoundModel using:
 *  - OpenAIChatCompletionsDriver (from @magnitudedev/drivers)
 *  - NativeChatCompletionsCodec  (from @magnitudedev/codecs)
 *
 * The codec and driver are paired into a NativeTransport at construction —
 * the (W, C) generics are inferred from the concrete codec+driver and
 * never escape this function.
 *
 */

import { Effect, Layer } from 'effect'
import { ProviderCatalog, ProviderState } from '@magnitudedev/providers'
import { NativeModelResolver, NativeModelNotConfigured } from './native-model-resolver'
import { makeNativeTransport, type NativeBoundModel } from './native-bound-model'
import { OpenAIChatCompletionsDriver } from '@magnitudedev/drivers'
import { NativeChatCompletionsCodec } from '@magnitudedev/codecs'

// =============================================================================
// Live Layer
// =============================================================================

export const NativeModelResolverLive = Layer.effect(
  NativeModelResolver,
  Effect.gen(function* () {
    const state   = yield* ProviderState
    const catalog = yield* ProviderCatalog

    return {
      resolve: (slot: string) =>
        Effect.gen(function* () {
          // 1. Get model selection for this slot
          const slotData = yield* state.peek(slot)
          if (!slotData) {
            return yield* Effect.fail(new NativeModelNotConfigured({ slot }))
          }

          const { model } = slotData

          // Resolve auth: prefer stored auth, then env
          let auth = slotData.auth
          if (!auth) {
            const provider = yield* catalog.getProvider(model.providerId)
            if (!provider) {
              return yield* Effect.fail(new NativeModelNotConfigured({ slot }))
            }
            // Try env keys from authMethods
            for (const method of provider.authMethods) {
              if (method.type === 'api-key' && method.envKeys) {
                for (const envKey of method.envKeys) {
                  const val = process.env[envKey]
                  if (val) {
                    auth = { type: 'api', key: val }
                    break
                  }
                }
              }
              if (auth) break
            }
          }

          if (!auth) {
            return yield* Effect.fail(new NativeModelNotConfigured({ slot }))
          }

          // 4. Get provider base URL
          const provider = yield* catalog.getProvider(model.providerId)
          const endpoint = provider?.defaultBaseUrl ?? 'https://api.fireworks.ai/inference/v1'

          // 5. Build codec + driver, then bundle them.
          //    The (W, C) generics are inferred here from the concrete
          //    NativeChatCompletionsCodec / OpenAIChatCompletionsDriver
          //    types and remain inside the makeNativeTransport closure.
          const wireModelName     = model.id
          const defaultMaxTokens  = model.maxOutputTokens ?? 32_768
          const supportsReasoning = model.supportsReasoning
          const supportsVision    = model.supportsVision

          const codec  = NativeChatCompletionsCodec({
            wireModelName,
            defaultMaxTokens,
            supportsReasoning,
            supportsVision,
          })
          const driver = OpenAIChatCompletionsDriver

          const transport = makeNativeTransport(codec, driver)

          const boundModel: NativeBoundModel = {
            model,
            auth,
            transport,
            wireConfig: {
              endpoint,
              wireModelName,
              defaultMaxTokens,
            },
          }

          return boundModel
        }),
    }
  }),
)
