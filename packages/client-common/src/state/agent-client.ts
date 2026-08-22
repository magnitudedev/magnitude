/**
 * The connection's Effect Query client over the ACN boundary.
 *
 * Every client↔ACN interaction is a declared query, mutation, or subscription
 * materialized through this client. RPC-backed implementations are derived
 * once per connection; domain services
 * and the change drain are installed as Layers in the same runtime.
 */
import type { RpcClient } from "@effect/rpc"
import { Layer } from "effect"
import { Client } from "@magnitudedev/effect-query"
import { AcnBoundary, AcnRpc } from "@magnitudedev/sdk"
import {
  clientServicesLayer,
  type ClientServices,
  type ClientServicesOptions,
} from "./client-services"

const acnImplementationsLayer = (
  protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>,
) => AcnRpc.layer(AcnBoundary).pipe(Layer.provide(protocolLayer))

export type AcnClientRequirements = Layer.Layer.Success<ReturnType<typeof acnImplementationsLayer>>

export type AgentClient = Client.Client<AcnClientRequirements | ClientServices, never>

export function createAgentClient(
  protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>,
  options: ClientServicesOptions = {},
): AgentClient {
  return Client.make<AcnClientRequirements, never, ClientServices, never>(
    acnImplementationsLayer(protocolLayer),
    (client) => clientServicesLayer(client, options),
  )
}
