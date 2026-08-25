/**
 * The connection's Effect Query client for the composed Magnitude boundary.
 *
 * Every first-party application operation uses ACN RPC. The serving-only
 * inference proxy is not part of this client. Every domain group's operations
 * are materialized at its name and share one runtime, QueryClient, registry,
 * and ACN change drain.
 */
import type { RpcClient } from "@effect/rpc"
import { Layer } from "effect"
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
) => magnitudeImplementationsLayer(protocolLayer)

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
