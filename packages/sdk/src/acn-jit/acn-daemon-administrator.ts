import { Context, Effect } from "effect"
import type { AcnAdministrationFailed } from "./errors"

/** Privileged local administration of the durable ACN assignment. */
export interface AcnDaemonAdministrator {
  readonly stopCurrent: Effect.Effect<void, AcnAdministrationFailed>
}

export const AcnDaemonAdministrator = Context.GenericTag<AcnDaemonAdministrator>(
  "@magnitudedev/sdk/AcnDaemonAdministrator",
)
