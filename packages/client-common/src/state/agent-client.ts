/** Shared ACN transport with AtomRpc and Effect Query materializations. */
import { Atom, AtomRpc } from "@effect-atom/atom-react"
import { RpcClient } from "@effect/rpc"
import { Layer } from "effect"
import { Client as EffectQueryClient } from "@magnitudedev/effect-query"
import { AcnRpcClientTag, MagnitudeRpcs } from "@magnitudedev/sdk"

export type AgentClientInstance = ReturnType<typeof createAgentClient>
export type AgentClient = AgentClientInstance

class AcnAtomRpcClient {}

/**
 * Create one flat RPC service. AtomRpc and Effect Query share that service;
 * each domain chooses one state system and never mixes both for the same data.
 */
export function createAgentClient(
  protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>,
) {
  const client = AtomRpc.Tag<AcnAtomRpcClient>()("AcnRpc", {
    group: MagnitudeRpcs,
    protocol: protocolLayer,
  })
  Atom.runtime.addGlobalLayer(client.layer)
  const rpcLayer = Layer.effect(AcnRpcClientTag, client).pipe(Layer.provide(client.layer))
  const effectQuery = EffectQueryClient.make(rpcLayer)
  return Object.assign(client, { effectQuery })
}
