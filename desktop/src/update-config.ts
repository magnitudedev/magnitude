import { decodePublisherPublicKey } from "@magnitudedev/release/hosted-update"
import { Effect, Schema } from "effect"

declare const __MAGNITUDE_UPDATE_CONFIGURATION__: unknown

const Configuration = Schema.Struct({
  origin: Schema.String,
  storageOrigin: Schema.String,
  keyId: Schema.String,
  publicKey: Schema.String,
  acceptance: Schema.Boolean,
}).pipe(Schema.filter(config => !config.acceptance || config.origin !== "https://magnitude.dev"))

/** Acceptance trust is an explicit build input, never a runtime environment override. */
export const readUpdateConfiguration = Schema.decodeUnknown(Configuration)(__MAGNITUDE_UPDATE_CONFIGURATION__).pipe(
  Effect.flatMap(config => decodePublisherPublicKey(config.publicKey).pipe(Effect.map(key => ({
    ...config, trustedPublishers: new Map([[config.keyId, key]]),
  })))),
)
