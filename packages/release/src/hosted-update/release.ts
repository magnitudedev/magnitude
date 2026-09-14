import { sign, verify, type KeyObject } from "node:crypto"
import { Effect, Schema } from "effect"
import { isNewerVersion, isValidVersion, admittedChannels, releaseChannelOf } from "../client-update/release-channels"

export const ReleaseTarget = Schema.Union(
  Schema.Struct({ os: Schema.Literal("darwin"), arch: Schema.Literal("arm64", "x64"), package: Schema.Literal("mac-zip", "dmg") }),
  Schema.Struct({ os: Schema.Literal("windows"), arch: Schema.Literal("x64"), package: Schema.Literal("windows-exe") }),
  Schema.Struct({ os: Schema.Literal("linux"), arch: Schema.Literal("arm64", "x64"), package: Schema.Literal("deb", "rpm") }),
)
export type ReleaseTarget = typeof ReleaseTarget.Type

const ReleaseContent = Schema.Struct({
  version: Schema.String.pipe(Schema.maxLength(96), Schema.pattern(/^[a-zA-Z0-9.+-]+$/), Schema.filter(isValidVersion)),
  bytes: Schema.Number.pipe(Schema.int(), Schema.positive(), Schema.filter(Number.isSafeInteger)),
  sha256: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/)),
})

/** Public response and durable publisher proof; no publication coordinates cross this boundary. */
export const UpdateRelease = Schema.Struct({
  ...ReleaseContent.fields,
  signature: Schema.String.pipe(Schema.pattern(/^[A-Za-z0-9+/]{86}==$/)),
})
export type UpdateRelease = typeof UpdateRelease.Type

export class InvalidUpdateRelease extends Schema.TaggedError<InvalidUpdateRelease>()("InvalidUpdateRelease", {}) {}
export class UpdateReleaseSigningFailed extends Schema.TaggedError<UpdateReleaseSigningFailed>()("UpdateReleaseSigningFailed", {}) {}

// Every field excludes newlines. Order, decimal bytes and the trailing newline are part of the protocol.
const signingBytes = (release: typeof ReleaseContent.Type, target: ReleaseTarget) => Buffer.from(
  `magnitude-update-release-v1\n${target.os}\n${target.arch}\n${target.package}\n${release.version}\n${release.bytes}\n${release.sha256}\n`, "utf8",
)

export const signUpdateRelease = (content: typeof ReleaseContent.Type, target: ReleaseTarget, privateKey: KeyObject) => Effect.gen(function* () {
  const decoded = yield* Schema.decodeUnknown(ReleaseContent)(content, { onExcessProperty: "error" })
  const destination = yield* Schema.decodeUnknown(ReleaseTarget)(target, { onExcessProperty: "error" })
  return yield* Effect.try(() => {
    if (privateKey.asymmetricKeyType !== "ed25519") throw new Error("Expected Ed25519 publisher")
    return UpdateRelease.make({ ...decoded, signature: sign(null, signingBytes(decoded, destination), privateKey).toString("base64") })
  })
}).pipe(Effect.mapError(() => new UpdateReleaseSigningFailed()))

/** Trust is bundled by the caller. An offer cannot choose or introduce a key. */
export const verifyUpdateRelease = (input: unknown, target: ReleaseTarget, trustedKeys: ReadonlyMap<string, KeyObject>) => Effect.gen(function* () {
  const release = yield* Schema.decodeUnknown(UpdateRelease)(input, { onExcessProperty: "error" })
  const destination = yield* Schema.decodeUnknown(ReleaseTarget)(target, { onExcessProperty: "error" })
  yield* Effect.try(() => {
    const signature = Buffer.from(release.signature, "base64")
    if (signature.length !== 64 || signature.toString("base64") !== release.signature) throw new Error("Invalid signature encoding")
    const bytes = signingBytes(release, destination)
    if (![...trustedKeys.values()].some(key => key.asymmetricKeyType === "ed25519" && verify(null, bytes, key, signature))) {
      throw new Error("Invalid publisher signature")
    }
  })
  return release
}).pipe(Effect.mapError(() => new InvalidUpdateRelease()))

export const acceptsUpdateRelease = (release: UpdateRelease, installedVersion: string): boolean =>
  isNewerVersion(release.version, installedVersion)
  && admittedChannels(releaseChannelOf(installedVersion)).has(releaseChannelOf(release.version))

export const updateInstallerFilename = (target: ReleaseTarget): string => {
  switch (target.package) {
    case "mac-zip": return "magnitude.zip"
    case "windows-exe": return "magnitude-setup.exe"
    default: return `magnitude.${target.package}`
  }
}
