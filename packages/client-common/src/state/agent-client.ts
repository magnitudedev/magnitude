/**
 * AgentClient — AtomRpc tag for the MagnitudeRpcs group.
 * Spec §6.2.
 *
 * Uses the SDK's recovering protocol layer with a host-provided
 * daemon discovery and launch. The client runtime owns endpoint selection and recovery;
 * platform process access only queries or starts an ACN.
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
 * display controller, file-watch, session-statuses — share one client
 * lifecycle and recovery authority. Each typed RPC client still builds its
 * own single-consumer protocol receiver.
 */
export function createAgentClient(
  protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>,
) {
  const client = AtomRpc.Tag<AgentClient>()("AgentClient", {
    group: MagnitudeRpcs,
    protocol: protocolLayer,
  })
  Atom.runtime.addGlobalLayer(client.layer)
  return client
}
