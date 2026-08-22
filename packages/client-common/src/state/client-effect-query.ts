import { Context } from "effect"
import type { Client } from "@magnitudedev/effect-query"
import type { AcnClientRequirements } from "./agent-client"

/** The connection's Effect Query client, as seen by domain services. */
export interface ClientEffectQuery
  extends Pick<Client.Client<AcnClientRequirements, never>, "query" | "mutation" | "subscription"> {}

export const ClientEffectQuery = Context.GenericTag<ClientEffectQuery>("client/EffectQuery")
