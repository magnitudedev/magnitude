import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { UpdateConfiguration } from "@magnitudedev/release/hosted-update"
import { Option, Schema } from "effect"
import { Digest } from "./domain"

/** Explicit updater-functionality fixtures, built from the same submitted source as the app.
 * They do not claim compatibility with an earlier released implementation.
 * Only the private authority's content address is public in reports. */
export const UpdateAcceptance = Schema.Struct({ sourceDigest: Digest, configuration: UpdateConfiguration.pipe(Schema.filter(config => config.acceptance && /^https:\/\/127\.0\.0\.1:[1-9][0-9]{3,4}$/.test(config.origin)
    && Option.exists(config.artifactDelivery, delivery => delivery._tag === "PrivateAcceptance" && delivery.origin === config.origin))),
  authority: Schema.Struct({ sha256: Digest, bytes: Schema.Int.pipe(Schema.between(1, 64 * 1024)) }),
  previous: ReleaseManifestSchema, candidate: ReleaseManifestSchema,
})
export type UpdateAcceptance = typeof UpdateAcceptance.Type
