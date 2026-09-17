import { createPrivateKey, createPublicKey, type KeyObject } from "node:crypto"
import { Effect, Schema } from "effect"
import { isNewerVersion, isValidVersion, admittedChannels, releaseChannelOf } from "../client-update/release-channels"
import type { UpdateRequest } from "./request"
import { ReleaseTarget, UpdateRelease, signUpdateRelease, verifyUpdateRelease } from "./release"

const Version = Schema.String.pipe(Schema.maxLength(96), Schema.filter(isValidVersion))
export const PublisherKeyId = Schema.String.pipe(Schema.pattern(/^[a-zA-Z0-9_-]{1,64}$/), Schema.brand("PublisherKeyId"))
const Digest = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/))
export const ArtifactId = Schema.String.pipe(Schema.pattern(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/), Schema.brand("UpdateArtifactId"))
export type ArtifactId = typeof ArtifactId.Type
const ArtifactFilename = Schema.String.pipe(Schema.maxLength(255), Schema.pattern(/^[a-zA-Z0-9][a-zA-Z0-9._~-]*$/))
const ReleaseTag = Schema.String.pipe(Schema.maxLength(256), Schema.pattern(/^(?:@magnitudedev\/cli@|desktop-update-acceptance\/[a-f0-9]{40}\/)[a-zA-Z0-9][a-zA-Z0-9._-]*$/))
export const ArtifactTarget = ReleaseTarget
export const UpdateManifest = Schema.Struct({
  protocol: Schema.Literal(1),
  version: Version,
  tag: ReleaseTag,
  commit: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{40}$/)),
  artifact: Schema.Struct({
    id: ArtifactId,
    target: ArtifactTarget,
    filename: ArtifactFilename,
    bytes: Schema.Number.pipe(Schema.int(), Schema.positive(), Schema.filter(Number.isSafeInteger)),
    sha256: Digest,
  }),
}).pipe(Schema.filter(manifest => manifest.tag === `@magnitudedev/cli@${manifest.version}` || manifest.tag === `desktop-update-acceptance/${manifest.commit}/${manifest.version}`))
export type UpdateManifest = typeof UpdateManifest.Type
/** Publisher-only metadata. HTTP handlers expose only release, never these coordinates. */
export const PublishedUpdate = Schema.Struct({ manifest: UpdateManifest, release: UpdateRelease })
export type PublishedUpdate = typeof PublishedUpdate.Type
export class InvalidUpdateManifest extends Schema.TaggedError<InvalidUpdateManifest>()("InvalidUpdateManifest", {}) {}
export class ReleaseSigningFailed extends Schema.TaggedError<ReleaseSigningFailed>()("ReleaseSigningFailed", {}) {}

export const signUpdateManifest = (manifest: UpdateManifest, privateKey: KeyObject) => Effect.gen(function* () {
  const decoded = yield* Schema.decodeUnknown(UpdateManifest)(manifest, { onExcessProperty: "error" })
  const release = yield* signUpdateRelease({ version: decoded.version, bytes: decoded.artifact.bytes, sha256: decoded.artifact.sha256 }, decoded.artifact.target, privateKey)
  return PublishedUpdate.make({ manifest: decoded, release })
}).pipe(Effect.mapError(() => new ReleaseSigningFailed()))

/** Verify stored content against bundled trust before serving the accepted publication. */
export const verifyUpdateManifest = (input: unknown, trustedKeys: ReadonlyMap<string, KeyObject>) => Effect.gen(function* () {
  const published = yield* Schema.decodeUnknown(PublishedUpdate)(input, { onExcessProperty: "error" })
  const { manifest } = published
  const release = yield* verifyUpdateRelease(published.release, manifest.artifact.target, trustedKeys)
  if (release.version !== manifest.version || release.bytes !== manifest.artifact.bytes || release.sha256 !== manifest.artifact.sha256) {
    return yield* new InvalidUpdateManifest()
  }
  return manifest
}).pipe(Effect.mapError(() => new InvalidUpdateManifest()))

export const acceptsUpdateManifest = (manifest: UpdateManifest, request: Pick<UpdateRequest, "version" | "os" | "arch" | "package">): boolean =>
  isNewerVersion(manifest.version, request.version)
  && admittedChannels(releaseChannelOf(request.version)).has(releaseChannelOf(manifest.version))
  && manifest.artifact.target.os === request.os && manifest.artifact.target.arch === request.arch
  && manifest.artifact.target.package === request.package

export const decodePublisherPrivateKey = (pem: string) => Effect.try({
  try: () => { const key = createPrivateKey(pem); if (key.asymmetricKeyType !== "ed25519") throw new Error("Wrong key"); return key },
  catch: () => new ReleaseSigningFailed(),
})
export const decodePublisherPublicKey = (pem: string) => Effect.try({
  try: () => { const key = createPublicKey(pem); if (key.asymmetricKeyType !== "ed25519") throw new Error("Wrong key"); return key },
  catch: () => new InvalidUpdateManifest(),
})
