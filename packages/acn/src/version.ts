import {
  ACN_BUILD_KIND as GENERATED_ACN_BUILD_KIND,
  ACN_COORDINATION_REVISION,
  ACN_DEVELOPMENT_KEY,
  MAGNITUDE_VERSION,
} from "@magnitudedev/version"
import {
  AcnIdentitySchema,
  AcnRevisionSchema,
  type AcnTarget,
} from "@magnitudedev/acn-protocol"

/**
 * ACN version, overridable via `MAGNITUDE_ACN_VERSION` env var for dev/testing.
 * Lets development clients and their candidate ACN use one explicit identity.
 */
export const ACN_VERSION = AcnIdentitySchema.make(
  process.env.MAGNITUDE_ACN_VERSION ?? MAGNITUDE_VERSION,
)

export const ACN_REVISION = AcnRevisionSchema.make(ACN_COORDINATION_REVISION)
export const ACN_BUILD_KIND: "published" | "development" = GENERATED_ACN_BUILD_KIND
export { ACN_DEVELOPMENT_KEY }

export const ACN_TARGET: AcnTarget = {
  identity: ACN_VERSION,
  revision: ACN_REVISION,
}
