import { Context } from "effect"
import type { Client } from "@magnitudedev/effect-query"
import type { MagnitudeBoundary, MagnitudeImplementationError } from "@magnitudedev/sdk"
import type { AcnClientRequirements } from "./agent-client"

/** The ACN boundary as seen by domain services: every operation materialized at its name. */
export interface ClientEffectQuery
  extends Client.Materialized<
    typeof MagnitudeBoundary,
    AcnClientRequirements,
    MagnitudeImplementationError
  > {}

export const ClientEffectQuery = Context.GenericTag<ClientEffectQuery>("client/EffectQuery")
