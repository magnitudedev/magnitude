import {
  ACN_COORDINATION_REVISION,
  MAGNITUDE_VERSION,
} from "@magnitudedev/version"

import {
  AcnIdentitySchema,
  AcnRevisionSchema,
  type AcnTarget,
} from "@magnitudedev/acn-protocol"


/**
 * SDK version, overridable via `MAGNITUDE_ACN_VERSION` env var for dev/testing.
 * This is only the client's initial ACN identity. `AcnConnection` owns the
 * effective identity after construction and advances it when the client adopts
 * a newer ACN.
 */
export const DAEMON_VERSION = AcnIdentitySchema.make(
  process.env.MAGNITUDE_ACN_VERSION ?? MAGNITUDE_VERSION,
)


export const DAEMON_REVISION = AcnRevisionSchema.make(ACN_COORDINATION_REVISION)


export const DAEMON_TARGET: AcnTarget = {
  identity: DAEMON_VERSION,
  revision: DAEMON_REVISION,
}
