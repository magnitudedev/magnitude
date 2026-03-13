/**
 * BrowserService
 *
 * Effect service that manages WebHarness instances per fork.
 * Lazy creation: harness is created on first get(), reused on subsequent calls.
 * Cleanup: release() stops harness and closes context.
 */

import { Context, Effect, Layer } from 'effect'
import { WebHarness, BrowserProvider, type WebHarnessOptions } from '@magnitudedev/browser-harness'
import { ProviderState } from '@magnitudedev/providers'

export interface BrowserServiceShape {
  readonly get: (forkId: string) => Effect.Effect<WebHarness>
  readonly release: (forkId: string) => Effect.Effect<void>
}

export class BrowserService extends Context.Tag('BrowserService')<
  BrowserService,
  BrowserServiceShape
>() {}

export const BrowserServiceLive = Layer.scoped(
  BrowserService,
  Effect.gen(function* () {
    const providerState = yield* ProviderState
    const harnesses = new Map<string, WebHarness>()

    yield* Effect.addFinalizer(() =>
      Effect.promise(async () => {
        for (const [_, harness] of harnesses) {
          harness.stop()
          await harness.context.close()
        }
        harnesses.clear()
      })
    )

    return {
      get: (forkId) => Effect.gen(function* () {
        const existing = harnesses.get(forkId)
        if (existing) return existing

        return yield* Effect.promise(async () => {
          const provider = BrowserProvider.getInstance()
          const context = await provider.newContext()

          // Gemini models use a 1000x1000 coordinate grid; set virtualScreenDimensions
          // so the harness transforms coordinates to the actual viewport
          const browserModel = await Effect.runPromise(providerState.peek('browser'))
          const isGemini = browserModel?.model.providerId === 'google' || browserModel?.model.providerId === 'google-vertex'
          const harnessOptions: WebHarnessOptions = isGemini
            ? { virtualScreenDimensions: { width: 1000, height: 1000 } }
            : {}

          const harness = new WebHarness(context, harnessOptions)
          await harness.start()
          harnesses.set(forkId, harness)
          return harness
        })
      }),

      release: (forkId) => Effect.promise(async () => {
        const harness = harnesses.get(forkId)
        if (harness) {
          harness.stop()
          await harness.context.close()
          harnesses.delete(forkId)
        }
      }),
    }
  })
)
