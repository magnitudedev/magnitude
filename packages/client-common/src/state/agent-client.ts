/** Shared ACN transport with AtomRpc and Effect Query materializations. */
import { Atom, AtomRpc, Registry } from "@effect-atom/atom-react"
import * as Reactivity from "@effect/experimental/Reactivity"
import { RpcClient } from "@effect/rpc"
import { Effect, Layer, Stream } from "effect"
import { Client as EffectQueryClient, Subscription } from "@magnitudedev/effect-query"
import {
  Acn,
  LocalInferenceHardwareMirror,
  MagnitudeRpcs,
  ProviderModelCatalogMirror,
  StreamChanges,
  projectChangeQueries,
  sessionChangeQueries,
  type AcnTransport,
} from "@magnitudedev/sdk"
import {
  clientServicesLayer,
  type ClientServices,
  type ClientServicesOptions,
} from "./client-services"

export type AgentClientInstance = ReturnType<typeof createAgentClient>
export type AgentClient = AgentClientInstance

class AcnAtomRpcClient {}

/**
 * Domains still materialized by AtomRpc observe pokes through Reactivity keys.
 * This mapping exists only until those domains move to the contract.
 */
const reactivityKeysFor = (query: string): ReadonlyArray<string> => {
  if (query === LocalInferenceHardwareMirror.id || query === ProviderModelCatalogMirror.id) return [query]
  if (projectChangeQueries.includes(query)) return ["projects"]
  if (sessionChangeQueries.includes(query)) return ["sessions"]
  return []
}

const allReactivityKeys: ReadonlyArray<string> = [
  LocalInferenceHardwareMirror.id,
  ProviderModelCatalogMirror.id,
  "projects",
  "sessions",
]

/**
 * Create one flat RPC service. AtomRpc and Effect Query share that transport;
 * each domain chooses one state system and never mixes both for the same data.
 */
export function createAgentClient(
  protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>,
  options: ClientServicesOptions = {},
) {
  const runtime = Atom.context({ memoMap: Effect.runSync(Layer.makeMemoMap) })
  const rpc = AtomRpc.Tag<AcnAtomRpcClient>()("AcnRpc", {
    group: MagnitudeRpcs,
    protocol: protocolLayer,
    runtime,
  })
  const transportLayer = Layer.effect(Acn.Client, Effect.map(rpc, (client) => Acn.transport(client))).pipe(
    Layer.provide(rpc.layer),
  )
  const effectQuery = EffectQueryClient.make<AcnTransport, never, ClientServices, never>(
    transportLayer,
    (client) => clientServicesLayer(client, options),
  )
  runtime.addGlobalLayer(Layer.scopedDiscard(Effect.gen(function* () {
    const reactivity = yield* Reactivity.Reactivity
    const registry = yield* Registry.AtomRegistry
    const changes = effectQuery.subscription(StreamChanges, {})
    yield* Effect.acquireRelease(
      Effect.sync(() => {
        let attempt = 0
        return registry.subscribe(changes, (state) => {
          if (state.attempt === attempt) return
          attempt = state.attempt
          if (attempt > 1) queueMicrotask(() => reactivity.unsafeInvalidate([...allReactivityKeys]))
        })
      }),
      (unsubscribe) => Effect.sync(unsubscribe),
    )
    yield* Subscription.events(changes).pipe(
      Stream.runForEach((change) => {
        const keys = reactivityKeysFor(change.query)
        return keys.length === 0 ? Effect.void : reactivity.invalidate([...keys])
      }),
      Effect.forkScoped,
    )
  })))
  return { rpc, effectQuery }
}
