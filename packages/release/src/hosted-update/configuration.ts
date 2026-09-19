import { Effect, Option, Schema } from "effect"
import { ArtifactDelivery, githubArtifactDelivery } from "./artifact-delivery"
import { decodePublisherPublicKey } from "./manifest"

export const UpdateConfiguration = Schema.Struct({
  origin: Schema.String.pipe(Schema.filter(value => {
    try { const url = new URL(value); return url.origin === value && url.protocol === "https:" }
    catch { return false }
  })),
  keyId: Schema.NonEmptyString,
  publicKey: Schema.NonEmptyString,
  acceptance: Schema.Boolean,
  windowsPublisher: Schema.optionalWith(Schema.NonEmptyString, { as: "Option", exact: true }),
  artifactDelivery: Schema.optionalWith(ArtifactDelivery, { as: "Option", exact: true }),
}).pipe(Schema.filter(config => {
  if (!config.acceptance) return Option.isNone(config.artifactDelivery) || config.artifactDelivery.value._tag === "Github"
  const hostname = new URL(config.origin).hostname
  return hostname !== "magnitude.dev" && !hostname.endsWith(".magnitude.dev")
}))

/** Both arguments originate in the compiled application, never runtime environment overrides. */
export const decodeUpdateConfiguration = (input: unknown, acceptanceBuild: boolean) => Schema.decodeUnknown(
  UpdateConfiguration.pipe(Schema.filter(config => config.acceptance === acceptanceBuild)),
)(input, { onExcessProperty: "error" }).pipe(
  Effect.flatMap(config => decodePublisherPublicKey(config.publicKey).pipe(Effect.map(key => ({
    ...config,
    artifactDelivery: Option.getOrElse(config.artifactDelivery, () => githubArtifactDelivery),
    trustedPublishers: new Map([[config.keyId, key]]),
  })))),
)
