/**
 * The connection's Effect Query client for the composed Magnitude boundary.
 *
 * ACN capabilities and client-common orchestration share one operation graph,
 * runtime, QueryClient, registry, and ACN change drain. The serving-only
 * inference proxy is not part of this client.
 */
import { Layer } from "effect"
import { Client } from "@magnitudedev/effect-query"
import { MagnitudeClient } from "@magnitudedev/sdk"
import {
  clientServicesLayer,
  type ClientServices,
  type ClientServicesOptions,
} from "./client-services"
import { MagnitudeOperations } from "./application-operations"

const acnImplementationsLayer = (client: MagnitudeClient) => Layer.succeed(MagnitudeClient, client)

export type AcnClientRequirements = Layer.Layer.Success<ReturnType<typeof acnImplementationsLayer>>

export type AgentClient = Client.GroupClient<
  typeof MagnitudeOperations,
  AcnClientRequirements | ClientServices,
  never
>

export function createAgentClient(
  client: MagnitudeClient,
  options: ClientServicesOptions = {},
): AgentClient {
  return Client.make<
    typeof MagnitudeOperations,
    AcnClientRequirements,
    never,
    ClientServices,
    never
  >(
    MagnitudeOperations,
    acnImplementationsLayer(client),
    (client) => clientServicesLayer(client, options),
  )
}
