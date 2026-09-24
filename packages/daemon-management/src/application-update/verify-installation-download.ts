import { FileSystem } from "@effect/platform"
import { InstallationChannel, verifyInstallationOffer, decodePublisherPublicKey } from "@magnitudedev/release/hosted-update"
import { Effect, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import { ApplicationUpdateFailed } from "./application-update"

/** Finite bootstrap verification; it neither acquires an owner nor runs the downloaded package. */
export const verifyWindowsInstallationDownload = (options: {
  readonly offer: string
  readonly artifact: string
  readonly channel: string
  readonly publicKey: string
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const chunks = yield* fs.stream(options.offer, { bytesToRead: 16385 }).pipe(Stream.runCollect)
  const bytes = Buffer.concat(Array.from(chunks))
  if (bytes.length > 16384) return yield* new ApplicationUpdateFailed({ message: "The installation offer is too large." })
  const input = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Unknown))(bytes.toString("utf8"))
  const channel = yield* Schema.decodeUnknown(InstallationChannel)(options.channel)
  const key = yield* decodePublisherPublicKey(options.publicKey)
  const offer = yield* verifyInstallationOffer(input, { os: "windows", arch: "x64", package: "windows-exe" }, channel, new Map([["publisher", key]]))
  const hash = yield* Effect.sync(() => createHash("sha256"))
  const size = yield* fs.stream(options.artifact).pipe(Stream.runFoldEffect(0, (count, chunk) => Effect.gen(function* () {
    if (count + chunk.length > offer.release.bytes) return yield* new ApplicationUpdateFailed({ message: "The installer size does not match its signed release." })
    yield* Effect.sync(() => hash.update(chunk))
    return count + chunk.length
  })))
  const digest = yield* Effect.sync(() => hash.digest("hex"))
  if (size !== offer.release.bytes || digest !== offer.release.sha256) {
    return yield* new ApplicationUpdateFailed({ message: "The installer does not match its signed release." })
  }
}).pipe(Effect.mapError(() => new ApplicationUpdateFailed({ message: "Application release verification failed. Download the installer again." })))
