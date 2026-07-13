import { MAGNITUDE_VERSION } from "@magnitudedev/version"

/**
 * SDK version, overridable via `MAGNITUDE_ACN_VERSION` env var for dev/testing.
 * Must match ACN_VERSION so client and daemon agree on the version segment
 * used for daemon registration/discovery.
 */
export const SDK_VERSION: string =
  process.env.MAGNITUDE_ACN_VERSION ?? MAGNITUDE_VERSION
