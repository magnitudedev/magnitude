/** Shared ACN transport with AtomRpc and Effect Query materializations. */
import { Atom, AtomRpc } from "@effect-atom/atom-react"
import * as Reactivity from "@effect/experimental/Reactivity"
import { RpcClient } from "@effect/rpc"
import { Cause, Duration, Effect, Layer, Schedule, Stream } from "effect"
import { Client as EffectQueryClient } from "@magnitudedev/effect-query"
import {
  AcnRpcClientTag,
  LocalInferenceHardwareMirror,
  MagnitudeRpcs,
  ProviderModelCatalogMirror,
} from "@magnitudedev/sdk"
import {
  clientServicesLayer,
  type ClientServices,
  type ClientServicesOptions,
} from "./client-services"
import { makeClientInvalidations } from "./client-invalidations"

export type AgentClientInstance = ReturnType<typeof createAgentClient>
export type AgentClient = AgentClientInstance

class AcnAtomRpcClient {}

const invalidationWatchReconnect = Schedule.exponential("100 millis").pipe(
  Schedule.modifyDelay((_, delay) => Duration.min(delay, Duration.seconds(5))),
  Schedule.jittered,
)

/**
 * Create one flat RPC service. AtomRpc and Effect Query share that service;
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
  const directMirrorIds = [
    LocalInferenceHardwareMirror.id,
    ProviderModelCatalogMirror.id,
  ]
  const directMirrorIdSet = new Set<string>(directMirrorIds)
  const invalidations = Effect.runSync(makeClientInvalidations)
  runtime.addGlobalLayer(Layer.scopedDiscard(Effect.gen(function* () {
    const client = yield* rpc
    const reactivity = yield* Reactivity.Reactivity
    yield* invalidations.events.pipe(
      Stream.runForEach((event) => {
        switch (event._tag) {
          case "Connected":
            return reactivity.invalidate([...directMirrorIds, "projects", "sessions"])
          case "MirroredState":
            return directMirrorIdSet.has(event.invalidation.id)
              ? reactivity.invalidate([event.invalidation.id])
              : Effect.void
          case "Projects":
            return reactivity.invalidate(["projects"])
          case "Sessions":
            return reactivity.invalidate(["sessions"])
        }
      }),
      Effect.forkScoped,
    )
    yield* Stream.unwrap(invalidations.publish({ _tag: "Connected" }).pipe(
      Effect.as(client("StreamClientInvalidations", {})),
    )).pipe(
      Stream.runForEach(invalidations.publish),
      Effect.tapErrorCause((cause) => Cause.isInterruptedOnly(cause)
        ? Effect.void
        : Effect.logWarning("Client invalidation watch disconnected; retrying").pipe(
            Effect.annotateLogs({ cause: Cause.pretty(cause).slice(0, 1_000) }),
          )),
      Effect.retry(invalidationWatchReconnect),
      Effect.catchAllCause((cause) => Cause.isInterruptedOnly(cause)
        ? Effect.void
        : Effect.logError(Cause.pretty(cause))),
      Effect.forkScoped,
    )
  })).pipe(Layer.provide(rpc.layer)))
  const rpcLayer = Layer.effect(AcnRpcClientTag, rpc).pipe(Layer.provide(rpc.layer))
  const effectQuery = EffectQueryClient.make<AcnRpcClientTag, never, ClientServices, never>(
    rpcLayer,
    (client) => clientServicesLayer(client, invalidations, options),
  )
  return { rpc, effectQuery }
}
