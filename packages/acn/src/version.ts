import { MAGNITUDE_VERSION } from "@magnitudedev/version"

/**
 * ACN version, overridable via `MAGNITUDE_ACN_VERSION` env var for dev/testing.
 * Lets multiple dev sessions share the same daemon by forcing them to the
 * same version segment.
 */
export const ACN_VERSION: string =
  process.env.MAGNITUDE_ACN_VERSION ?? MAGNITUDE_VERSION
