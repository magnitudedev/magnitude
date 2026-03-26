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
import { makeProviderRuntimeLive, type ProviderRuntime } from './live'

type ProviderRuntimeServices = ProviderCatalog | ProviderState | AppConfig | ProviderAuth

/** Maps an Effect service shape to a Promise-based facade */
type Promisify<S> = {
  [K in keyof S]: S[K] extends (...args: infer A) => Effect.Effect<infer R, infer _E, infer _R>
    ? (...args: A) => Promise<R>
    : never
}

export interface ProviderClient<TSlot extends string = string> {
  readonly catalog: Promisify<ProviderCatalogShape>
  readonly state: Promisify<ProviderStateShape<TSlot>>
  readonly config: Promisify<AppConfigShape<TSlot>>
  readonly auth: Promisify<ProviderAuthShape>
  readonly layer: ProviderRuntime<TSlot>
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

export async function createProviderClient<TSlot extends string>(args: { slots: readonly TSlot[] }): Promise<ProviderClient<TSlot>> {
  const layer = makeProviderRuntimeLive<TSlot>()
  const runtime = ManagedRuntime.make(layer)

  await runtime.runPromise(bootstrapProviderRuntime({ slots: args.slots }))

  return {
    catalog: buildServiceFacade(ProviderCatalog, (e) => runtime.runPromise(e)),
    state: buildServiceFacade(ProviderState, (e) => runtime.runPromise(e)) as Promisify<ProviderStateShape<TSlot>>,
    config: buildServiceFacade(AppConfig, (e) => runtime.runPromise(e)),
    auth: buildServiceFacade(ProviderAuth, (e) => runtime.runPromise(e)),
    layer,
  }
}