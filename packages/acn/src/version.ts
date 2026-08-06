import { MAGNITUDE_VERSION } from "@magnitudedev/version"
import { AcnIdentitySchema } from "@magnitudedev/acn-protocol"

/**
 * ACN version, overridable via `MAGNITUDE_ACN_VERSION` env var for dev/testing.
 * Lets development clients and their candidate ACN use one explicit identity.
 */
export const ACN_VERSION = AcnIdentitySchema.make(
  process.env.MAGNITUDE_ACN_VERSION ?? MAGNITUDE_VERSION,
)
