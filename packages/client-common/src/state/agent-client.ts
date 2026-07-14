/**
 * AgentClient — AtomRpc tag for the MagnitudeRpcs group.
 * Spec §6.2.
 *
 * Uses the SDK's recovering protocol layer with a host-provided
 * `DaemonSpawner`. The spawner is the only daemon lifecycle contract; URLs
 * are transport details resolved inside the SDK protocol layer.
 */
import { AtomRpc, Atom } from "@effect-atom/atom-react"
import { RpcClient } from "@effect/rpc"
import type { Layer } from "effect"
import { MagnitudeRpcs } from "@magnitudedev/sdk"

/**
 * Placeholder class used as the type identifier for the AgentClient tag.
 */
export class AgentClient {}

export type AgentClientInstance = ReturnType<typeof createAgentClient>

/**
 * Create an AgentClient AtomRpc tag backed by a shared protocol layer.
 *
 * The protocol layer must be created once at startup (by the Platform) and
 * passed here. This ensures all RPC consumers — AtomRpc mutations, the
 * display controller, file-watch, session-statuses — share one resolver,
 * one endpoint cache, one transport.
 */
export function createAgentClient(protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>) {
  const tag = AtomRpc.Tag<AgentClient>()("AgentClient", {
    group: MagnitudeRpcs,
    protocol: protocolLayer,
  })

  // Register the tag's layer as a global layer so all atoms can access it
  Atom.runtime.addGlobalLayer(tag.layer)

  return tag
}
