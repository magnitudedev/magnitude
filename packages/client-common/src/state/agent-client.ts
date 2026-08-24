/**
 * The connection's Effect Query client for the composed Magnitude boundary.
 *
 * ACN application operations use RPC while inference operations use the stable
 * HTTP proxy. Every domain group's operations are materialized at its name;
 * both transports share one runtime, QueryClient, registry, and change drains.
 */
import type { RpcClient } from "@effect/rpc"
import { Layer } from "effect"
import { FetchHttpClient } from "@effect/platform"
import { Client } from "@magnitudedev/effect-query"
import {
  MagnitudeBoundary,
  type MagnitudeImplementationError,
  magnitudeImplementationsLayer,
} from "@magnitudedev/sdk"
import {
  clientServicesLayer,
  type ClientServices,
  type ClientServicesOptions,
} from "./client-services"

const acnImplementationsLayer = (
  protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>,
) => magnitudeImplementationsLayer(protocolLayer).pipe(
  Layer.provide(FetchHttpClient.layer),
)

export type AcnClientRequirements = Layer.Layer.Success<ReturnType<typeof acnImplementationsLayer>>

export type AgentClient = Client.GroupClient<
  typeof MagnitudeBoundary,
  AcnClientRequirements | ClientServices,
  MagnitudeImplementationError
>

export function createAgentClient(
  protocolLayer: Layer.Layer<RpcClient.Protocol, never, never>,
  options: ClientServicesOptions = {},
): AgentClient {
  return Client.make<
    typeof MagnitudeBoundary,
    AcnClientRequirements,
    MagnitudeImplementationError,
    ClientServices,
    never
  >(
    MagnitudeBoundary,
    acnImplementationsLayer(protocolLayer),
    (client) => clientServicesLayer(client, options),
  )
}
