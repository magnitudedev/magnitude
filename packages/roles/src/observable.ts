import { Effect, Layer } from 'effect'
import type { ObservableConfig, BoundObservable, ObservationPart } from './types'

/**
 * Create an observable configuration.
 * Observables capture environmental state (screenshots, etc.) before each agent turn.
 */
export function createObservable<R = never>(config: ObservableConfig<R>): ObservableConfig<R> {
  return config
}

/**
 * Bind an observable to a layer factory, satisfying its Effect requirements.
 * Same pattern as bindToolGroup — layer is created lazily and cached.
 */
export function bindObservable<R>(
  observable: ObservableConfig<R>,
  createLayer: () => Effect.Effect<Layer.Layer<R>>
): BoundObservable {
  let cachedLayer: Layer.Layer<R> | null = null
  return {
    name: observable.name,
    observe: (): Effect.Effect<ObservationPart[]> => Effect.gen(function* () {
      if (!cachedLayer) {
        cachedLayer = yield* createLayer()
      }
      return yield* observable.observe().pipe(
        Effect.provide(cachedLayer)
      )
    })
  }
}
