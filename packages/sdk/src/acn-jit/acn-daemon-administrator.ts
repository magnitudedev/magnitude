import { Context, Effect } from "effect"
import type { DaemonError } from "./errors"

/** Privileged local administration of the durable ACN assignment. */
export interface AcnDaemonAdministrator {
  readonly stopCurrent: Effect.Effect<void, DaemonError>
}

export const AcnDaemonAdministrator = Context.GenericTag<AcnDaemonAdministrator>(
  "@magnitudedev/sdk/AcnDaemonAdministrator",
)
