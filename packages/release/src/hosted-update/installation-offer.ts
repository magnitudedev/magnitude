import { Effect, Schema } from "effect"
import type { KeyObject } from "node:crypto"
import { admittedChannels, releaseChannelOf } from "../client-update/release-channels"
import { githubArtifactUrl, isGithubReleaseAssetUrl } from "./github-artifact"
import { verifyUpdateManifest, type PublishedUpdate } from "./manifest"
import { ReleaseTarget, UpdateRelease, verifyUpdateRelease } from "./release"

export const InstallationChannel = Schema.Literal("stable", "beta", "alpha")
export const InstallationOffer = Schema.Struct({
  release: UpdateRelease,
  download: Schema.String.pipe(Schema.maxLength(2048), Schema.filter(isGithubReleaseAssetUrl)),
})
export type InstallationOffer = typeof InstallationOffer.Type
export class InvalidInstallationOffer extends Schema.TaggedError<InvalidInstallationOffer>()("InvalidInstallationOffer", {}) {}

/** Installation has no prior version; platform and channel are explicit caller choices. */
export const verifyInstallationOffer = (input: unknown, target: ReleaseTarget,
  channel: typeof InstallationChannel.Type, trustedPublishers: ReadonlyMap<string, KeyObject>) => Effect.gen(function* () {
  const offer = yield* Schema.decodeUnknown(InstallationOffer)(input, { onExcessProperty: "error" })
  const selected = yield* Schema.decodeUnknown(InstallationChannel)(channel)
  yield* verifyUpdateRelease(offer.release, target, trustedPublishers)
  if (!admittedChannels(selected).has(releaseChannelOf(offer.release.version))) return yield* new InvalidInstallationOffer()
  return offer
}).pipe(Effect.mapError(() => new InvalidInstallationOffer()))

/** Reuse accepted publisher records without exposing their internal publication coordinates. */
export const installationOfferFromPublication = (published: PublishedUpdate, trustedPublishers: ReadonlyMap<string, KeyObject>) =>
  verifyUpdateManifest(published, trustedPublishers).pipe(Effect.map(manifest => InstallationOffer.make({
    release: published.release, download: githubArtifactUrl(manifest),
  })))
