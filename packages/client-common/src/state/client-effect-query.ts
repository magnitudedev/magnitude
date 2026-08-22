import { Context } from "effect"
import type { Client } from "@magnitudedev/effect-query"
import type { AcnTransport } from "@magnitudedev/sdk"

/** The connection's Effect Query client, as seen by domain services. */
export interface ClientEffectQuery
  extends Pick<Client.Client<AcnTransport, never>, "query" | "mutation" | "subscription"> {}

export const ClientEffectQuery = Context.GenericTag<ClientEffectQuery>("client/EffectQuery")
