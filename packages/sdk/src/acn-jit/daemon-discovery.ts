import {
  AcnEndpointSchema,
  AcnHealthStateSchema,
} from "@magnitudedev/acn-protocol"
import { Context, Effect, Option, Schema } from "effect"
import type { DaemonError } from "./errors"

/** The canonical daemon state observed through registration and health. */
export const DaemonStatusSchema = Schema.extend(
  AcnEndpointSchema,
  Schema.Struct({
    pid: Schema.Number.pipe(Schema.int(), Schema.positive()),
    state: AcnHealthStateSchema,
  }),
)
export type DaemonStatus = typeof DaemonStatusSchema.Type

/** Read-only access to the canonical Magnitude daemon. */
export interface DaemonDiscovery {
  readonly current: () => Effect.Effect<
    Option.Option<DaemonStatus>,
    DaemonError
  >
}

export const DaemonDiscovery = Context.GenericTag<DaemonDiscovery>(
  "@magnitudedev/sdk/DaemonDiscovery",
)
