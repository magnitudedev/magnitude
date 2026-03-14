import { Context, Effect, Layer, ManagedRuntime } from 'effect'
import { AppConfig, type AppConfigShape } from '@magnitudedev/storage'
import { bootstrapProviderRuntime } from './bootstrap'
import {
  ProviderAuth,
  ProviderCatalog,
  ProviderState,
  type ProviderAuthShape,
  type ProviderCatalogShape,
  type ProviderStateShape,
} from './contracts'
import { makeProviderRuntimeLive } from './live'

type ProviderRuntimeServices = ProviderCatalog | ProviderState | AppConfig | ProviderAuth

/** Maps an Effect service shape to a Promise-based facade */
type Promisify<S> = {
  [K in keyof S]: S[K] extends (...args: infer A) => Effect.Effect<infer R, infer _E, infer _R>
    ? (...args: A) => Promise<R>
    : never
}

export interface ProviderClient {
  readonly catalog: Promisify<ProviderCatalogShape>
  readonly state: Promisify<ProviderStateShape>
  readonly config: Promisify<AppConfigShape>
  readonly auth: Promisify<ProviderAuthShape>
  readonly layer: Layer.Layer<ProviderRuntimeServices>
}

// Internal: builds a Promise-based facade for a service tag using a proxy.
// The proxy body is dynamically typed; the return type is statically safe via Promisify<Shape>.
/* eslint-disable @typescript-eslint/no-explicit-any */
function buildServiceFacade<Id extends ProviderRuntimeServices, Shape>(
  tag: Context.Tag<Id, Shape>,
  run: (effect: Effect.Effect<any, any, ProviderRuntimeServices>) => Promise<any>,
): Promisify<Shape> {
  return new Proxy(Object.create(null) as Promisify<Shape>, {
    get(_target, prop: string) {
      return (...args: any[]) =>
        run(Effect.flatMap(tag, (svc: any) => svc[prop](...args)))
    },
  })
}
/* eslint-enable @typescript-eslint/no-explicit-any */

export async function createProviderClient(): Promise<ProviderClient> {
  const layer = makeProviderRuntimeLive()
  const runtime = ManagedRuntime.make(layer)

  await runtime.runPromise(bootstrapProviderRuntime)

  return {
    catalog: buildServiceFacade(ProviderCatalog, (e) => runtime.runPromise(e)),
    state: buildServiceFacade(ProviderState, (e) => runtime.runPromise(e)),
    config: buildServiceFacade(AppConfig, (e) => runtime.runPromise(e)),
    auth: buildServiceFacade(ProviderAuth, (e) => runtime.runPromise(e)),
    layer,
  }
}