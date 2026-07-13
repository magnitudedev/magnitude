/**
 * AgentClient — AtomRpc tag for the MagnitudeRpcs group.
 * Spec §6.2.
 *
 * Uses the SDK's recovering protocol layer with a host-provided
 * `DaemonSpawner`. The spawner is the only daemon lifecycle contract; URLs
 * are transport details resolved inside the SDK protocol layer.
 */
import { AtomRpc, Atom } from "@effect-atom/atom-react"
import { FetchHttpClient } from "@effect/platform"
import { Layer } from "effect"
import {
  MagnitudeRpcs,
  recoveringProtocolLayer,
  type DaemonSpawnerTag,
} from "@magnitudedev/sdk"

/**
 * Placeholder class used as the type identifier for the AgentClient tag.
 */
export class AgentClient {}

export type AgentClientInstance = ReturnType<typeof createAgentClient>

/**
 * Create an AgentClient AtomRpc tag backed by a `DaemonSpawner`.
 * Call this at renderer startup with the host's spawner layer, then pass
 * the result to Atom.runtime.addGlobalLayer(instance.layer).
 */
export function createAgentClient(daemonSpawnerLayer: Layer.Layer<DaemonSpawnerTag, never, never>) {
  const tag = AtomRpc.Tag<AgentClient>()("AgentClient", {
    group: MagnitudeRpcs,
    protocol: recoveringProtocolLayer().pipe(
      Layer.provide(Layer.mergeAll(FetchHttpClient.layer, daemonSpawnerLayer)),
    ),
  })

  // Register the tag's layer as a global layer so all atoms can access it
  Atom.runtime.addGlobalLayer(tag.layer)

  return tag
}
