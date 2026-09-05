import { Context } from "effect"
import type { Client } from "@magnitudedev/effect-query"
import type { AcnQueries } from "../operations"
import type { AcnClientRequirements } from "./agent-client"

/** The ACN boundary as seen by domain services: every operation materialized at its name. */
export interface ClientEffectQuery
  extends Client.Materialized<
    typeof AcnQueries,
    AcnClientRequirements,
    never
  > {}

export const ClientEffectQuery = Context.GenericTag<ClientEffectQuery>("client/EffectQuery")
