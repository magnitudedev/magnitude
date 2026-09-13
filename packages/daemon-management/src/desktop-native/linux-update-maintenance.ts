import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema, Stream } from "effect"
import { dirname, join, parse } from "node:path"
import { decodePublisherPublicKey } from "@magnitudedev/release/hosted-update"
import { LinuxPackageUpdate, LinuxPackageUpdateFailed, makeLinuxPackageInstaller } from "./linux-update-package"

const Trust = Schema.Struct({ keyId: Schema.NonEmptyString, publicKey: Schema.NonEmptyString })
const installedCli = "/usr/lib/magnitude-desktop/resources/magnitude"

/** A private entry of the installed CLI. No caller-provided key can authorize a package. */
export const installLinuxApplicationUpdate = (requestPath: string, currentVersion: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  if (process.platform !== "linux" || process.getuid?.() !== 0 || (yield* fs.realPath(process.execPath)) !== installedCli) {
    return yield* new LinuxPackageUpdateFailed({ message: "Run application updates through the installed Magnitude desktop." })
  }
  const callerUid = Number(process.env.PKEXEC_UID)
  if (!Number.isSafeInteger(callerUid) || callerUid <= 0) {
    return yield* new LinuxPackageUpdateFailed({ message: "The desktop authorization did not identify the requesting user." })
  }
  const trustPath = join(dirname(installedCli), "update-trust.json")
  for (let path = trustPath; ; path = dirname(path)) {
    const info = yield* fs.stat(path)
    if (Option.getOrUndefined(info.uid) !== 0 || (info.mode & 0o022) !== 0 || (yield* fs.realPath(path)) !== path) {
      return yield* new LinuxPackageUpdateFailed({ message: "The installed publisher trust is not protected by the system." })
    }
    if (path === parse(path).root) break
  }
  const trustInfo = yield* fs.stat(trustPath)
  const formatPath = join(dirname(installedCli), "update-package.json")
  const formatInfo = yield* fs.stat(formatPath)
  const requestInfo = yield* fs.stat(requestPath)
  if (trustInfo.type !== "File" || trustInfo.size > 16_384n || formatInfo.type !== "File" || formatInfo.size > 1024n
    || Option.getOrUndefined(formatInfo.uid) !== 0 || (formatInfo.mode & 0o022) !== 0 || (yield* fs.realPath(formatPath)) !== formatPath
    || requestInfo.type !== "File" || requestInfo.size > 32_768n
    || Option.getOrUndefined(requestInfo.uid) !== callerUid) {
    return yield* new LinuxPackageUpdateFailed({ message: "The application update request is invalid." })
  }
  const trust = yield* fs.readFileString(trustPath).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Trust))))
  const key = yield* decodePublisherPublicKey(trust.publicKey)
  const format = yield* fs.readFileString(formatPath).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ format: Schema.Literal("deb", "rpm") })))))
  const request = yield* fs.stream(requestPath, { bytesToRead: 32_769 }).pipe(Stream.runFold(Buffer.alloc(0), (all, bytes) => Buffer.concat([all, bytes])),
    Effect.flatMap(bytes => Effect.gen(function* () {
      if (bytes.length > 32_768) return yield* new LinuxPackageUpdateFailed({ message: "The update request exceeded its size limit." })
      return yield* Schema.decodeUnknown(Schema.parseJson(LinuxPackageUpdate))(bytes.toString("utf8"))
    })))
  const installer = yield* makeLinuxPackageInstaller({ currentVersion, package: format.format, callerUid, trustedPublishers: new Map([[trust.keyId, key]]) })
  yield* installer.install(request)
}).pipe(Effect.mapError(error => error instanceof LinuxPackageUpdateFailed ? error
  : new LinuxPackageUpdateFailed({ message: "Could not verify the installed publisher trust or update request." })))
