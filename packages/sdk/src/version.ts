import { MAGNITUDE_VERSION } from "@magnitudedev/version"
import { AcnIdentitySchema } from "@magnitudedev/acn-protocol"

/**
 * SDK version, overridable via `MAGNITUDE_ACN_VERSION` env var for dev/testing.
 * This is only the client's initial ACN identity. `AcnJitRuntime` owns the
 * effective identity after construction and advances it when the client adopts
 * a newer ACN.
 */
export const SDK_VERSION = AcnIdentitySchema.make(
  process.env.MAGNITUDE_ACN_VERSION ?? MAGNITUDE_VERSION,
)
