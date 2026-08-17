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
import { runMirroredStateInvalidationWatch } from "./mirrored-state-invalidation"

export type AgentClientInstance = ReturnType<typeof createAgentClient>
export type AgentClient = AgentClientInstance

class AcnAtomRpcClient {}

const projectWatchReconnect = Schedule.exponential("100 millis").pipe(
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
  runtime.addGlobalLayer(Layer.scopedDiscard(Effect.gen(function* () {
    const client = yield* rpc
    const reactivity = yield* Reactivity.Reactivity
    yield* runMirroredStateInvalidationWatch(
      client,
      () => reactivity.invalidate(directMirrorIds),
      (event) => reactivity.invalidate([event.id]),
    ).pipe(Effect.forkScoped)
    yield* client("StreamProjectChanges", {}).pipe(
      Stream.runForEach(() => reactivity.invalidate(["projects", "sessions"])),
      Effect.tapErrorCause((cause) => Cause.isInterruptedOnly(cause)
        ? Effect.void
        : Effect.logWarning("Project watch disconnected; retrying").pipe(
            Effect.annotateLogs({ cause: Cause.pretty(cause).slice(0, 1_000) }),
          )),
      Effect.retry(projectWatchReconnect),
      Effect.catchAllCause((cause) => Cause.isInterruptedOnly(cause)
        ? Effect.void
        : Effect.logError(Cause.pretty(cause))),
      Effect.forkScoped,
    )
  })).pipe(Layer.provide(rpc.layer)))
  const rpcLayer = Layer.effect(AcnRpcClientTag, rpc).pipe(Layer.provide(rpc.layer))
  const effectQuery = EffectQueryClient.make<AcnRpcClientTag, never, ClientServices, never>(
    rpcLayer,
    (client) => clientServicesLayer(client, options),
  )
  return { rpc, effectQuery }
}
