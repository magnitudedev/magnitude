/**
 * The connection's Effect Query client for the composed Magnitude boundary.
 *
 * ACN capabilities and client-common orchestration share one operation graph,
 * runtime, QueryClient, registry, and ACN change drain. The serving-only
 * inference proxy is not part of this client.
 */
import type { RpcClient } from "@effect/rpc"
import { Layer } from "effect"
import { Client } from "@magnitudedev/effect-query"
import {
  type MagnitudeImplementationError,
  magnitudeImplementationsLayer,
} from "@magnitudedev/sdk"
import {
  clientServicesLayer,
  type ClientServices,
  type ClientServicesOptions,
} from "./client-services"
import { MagnitudeOperations } from "./application-operations"

const acnImplementationsLayer = (
  protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>,
) => magnitudeImplementationsLayer(protocolLayer)

export type AcnClientRequirements = Layer.Layer.Success<ReturnType<typeof acnImplementationsLayer>>

export type AgentClient = Client.GroupClient<
  typeof MagnitudeOperations,
  AcnClientRequirements | ClientServices,
  MagnitudeImplementationError
>

export function createAgentClient(
  protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>,
  options: ClientServicesOptions = {},
): AgentClient {
  return Client.make<
    typeof MagnitudeOperations,
    AcnClientRequirements,
    MagnitudeImplementationError,
    ClientServices,
    never
  >(
    MagnitudeOperations,
    acnImplementationsLayer(protocolLayer),
    (client) => clientServicesLayer(client, options),
  )
}
