import { createPrivateKey, createPublicKey, sign, verify, type KeyObject } from "node:crypto"
import { Effect, Schema } from "effect"
import { isNewerVersion, isValidVersion, admittedChannels, releaseChannelOf } from "../client-update/release-channels"
import type { UpdateRequest } from "./request"

const Version = Schema.String.pipe(Schema.maxLength(96), Schema.filter(isValidVersion))
export const PublisherKeyId = Schema.String.pipe(Schema.pattern(/^[a-zA-Z0-9_-]{1,64}$/), Schema.brand("PublisherKeyId"))
const Digest = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/))
export const ArtifactId = Schema.String.pipe(Schema.pattern(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/), Schema.brand("UpdateArtifactId"))
export type ArtifactId = typeof ArtifactId.Type
const ArtifactPath = Schema.String.pipe(Schema.maxLength(512), Schema.pattern(/^releases\/[a-zA-Z0-9][a-zA-Z0-9._-]*\/[a-zA-Z0-9][a-zA-Z0-9._-]*$/))
export const ArtifactTarget = Schema.Union(
  Schema.Struct({ os: Schema.Literal("darwin"), arch: Schema.Literal("arm64", "x64"), package: Schema.Literal("mac-zip", "dmg") }),
  Schema.Struct({ os: Schema.Literal("windows"), arch: Schema.Literal("x64"), package: Schema.Literal("windows-exe") }),
  Schema.Struct({ os: Schema.Literal("linux"), arch: Schema.Literal("arm64", "x64"), package: Schema.Literal("deb", "rpm") }),
)
export const UpdateManifest = Schema.Struct({
  protocol: Schema.Literal(1),
  version: Version,
  commit: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{40}$/)),
  artifact: Schema.Struct({
    id: ArtifactId,
    target: ArtifactTarget,
    path: ArtifactPath,
    bytes: Schema.Number.pipe(Schema.int(), Schema.positive(), Schema.filter(Number.isSafeInteger)),
    sha256: Digest,
  }),
}).pipe(Schema.filter(manifest => manifest.artifact.path.startsWith(`releases/${manifest.version}/`)))
export type UpdateManifest = typeof UpdateManifest.Type
export const SignedUpdateManifest = Schema.Struct({
  keyId: PublisherKeyId,
  payload: Schema.String.pipe(Schema.maxLength(16384)),
  signature: Schema.String.pipe(Schema.maxLength(128)),
})
export type SignedUpdateManifest = typeof SignedUpdateManifest.Type
export class InvalidUpdateManifest extends Schema.TaggedError<InvalidUpdateManifest>()("InvalidUpdateManifest", {}) {}
export class ReleaseSigningFailed extends Schema.TaggedError<ReleaseSigningFailed>()("ReleaseSigningFailed", {}) {}
const context = "magnitude-release-v1\n"

export const signUpdateManifest = (manifest: UpdateManifest, keyId: typeof PublisherKeyId.Type, privateKey: KeyObject) =>
  Schema.encode(Schema.parseJson(UpdateManifest))(manifest).pipe(
    Effect.mapError(() => new ReleaseSigningFailed()),
    Effect.flatMap(json => Effect.try({ try: () => {
      if (privateKey.asymmetricKeyType !== "ed25519") throw new Error("Expected Ed25519 publisher")
      return SignedUpdateManifest.make({ keyId, payload: Buffer.from(json).toString("base64"),
        signature: sign(null, Buffer.from(context + json), privateKey).toString("base64") })
    }, catch: () => new ReleaseSigningFailed() })),
  )

/** Keys come only from application-embedded trust, never from the offer itself. */
export const verifyUpdateManifest = (input: unknown, trustedKeys: ReadonlyMap<string, KeyObject>) => Effect.gen(function* () {
  const envelope = yield* Schema.decodeUnknown(SignedUpdateManifest)(input, { onExcessProperty: "error" }).pipe(Effect.mapError(() => new InvalidUpdateManifest()))
  const json = yield* Effect.try({ try: () => {
    const key = trustedKeys.get(envelope.keyId)
    if (!key || key.asymmetricKeyType !== "ed25519") throw new Error("Unknown publisher")
    const payload = Buffer.from(envelope.payload, "base64"), signature = Buffer.from(envelope.signature, "base64")
    if (payload.toString("base64") !== envelope.payload || signature.toString("base64") !== envelope.signature || signature.length !== 64) throw new Error("Invalid envelope")
    if (!verify(null, Buffer.concat([Buffer.from(context), payload]), key, signature)) throw new Error("Invalid signature")
    return new TextDecoder("utf-8", { fatal: true }).decode(payload)
  }, catch: () => new InvalidUpdateManifest() })
  return yield* Schema.decodeUnknown(Schema.parseJson(UpdateManifest))(json, { onExcessProperty: "error" }).pipe(Effect.mapError(() => new InvalidUpdateManifest()))
})

export const acceptsUpdateManifest = (manifest: UpdateManifest, request: UpdateRequest): boolean =>
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
