import { FileSystem } from "@effect/platform"
import { Effect, Option, Ref, Schedule, Schema, Stream } from "effect"
import { createHash, randomUUID } from "node:crypto"
import { win32 } from "node:path"
import { isNewerVersion } from "@magnitudedev/release"
import { PrivateFilePermissions, WindowsInstallerVerifier, WindowsUpdateResult, startWindowsUpdateHandoff, type WindowsUpdateHandoffRequest } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateFailed } from "./application-update"
import { hostedUpdateSource, type HostedUpdateSourceOptions } from "./hosted-update-source"

export const makeWindowsUpdateSource = (options: HostedUpdateSourceOptions & {
  readonly stateDirectory: string
  readonly applicationPath: string
  readonly cliPath: string
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const permissions = yield* PrivateFilePermissions
  const verifier = yield* WindowsInstallerVerifier
  const prepared = yield* Ref.make(Option.none<Omit<WindowsUpdateHandoffRequest, "showWindow">>())
  const resultPath = win32.join(options.stateDirectory, "update-result.json")
  const prior = (yield* fs.exists(resultPath)) ? yield* Effect.gen(function* () {
    if ((yield* fs.stat(resultPath)).size > 32_768n) return yield* new ApplicationUpdateFailed({ message: "Invalid update result." })
    const result = yield* fs.readFileString(resultPath).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(WindowsUpdateResult))))
    if (result.request.stateDirectory !== options.stateDirectory || result.request.applicationPath !== options.applicationPath) {
      return yield* new ApplicationUpdateFailed({ message: "Unrelated update result." })
    }
    // Windows keeps the copied helper open until it exits after launching this desktop.
    yield* fs.remove(result.request.preparedDirectory, { recursive: true, force: true }).pipe(
      Effect.retry(Schedule.intersect(Schedule.spaced("100 millis"), Schedule.recurs(9))), Effect.ignore,
    )
    return isNewerVersion(result.request.version, options.metadata.version) ? result.error : Option.none<string>()
  }).pipe(Effect.catchAll(() => Effect.succeed(Option.some("The previous application update result could not be read. Check for updates to retry.")))) : Option.none<string>()
  const source = hostedUpdateSource(options, (archive, candidate) => Effect.gen(function* () {
    const parent = win32.join(options.stateDirectory, "application-updates")
    yield* permissions.prepareDirectory(parent)
    const directory = win32.join(parent, `prepared-${randomUUID()}`)
    yield* permissions.prepareDirectory(directory)
    yield* Effect.gen(function* () {
      const installer = win32.join(directory, "magnitude-setup.exe")
      yield* permissions.createFile(installer)
      const digest = createHash("sha256")
      let copied = 0
      yield* fs.stream(archive).pipe(Stream.tap(bytes => Effect.gen(function* () {
        copied += bytes.length
        if (copied > candidate.manifest.artifact.bytes) return yield* new ApplicationUpdateFailed({ message: "The downloaded installer changed during preparation." })
        digest.update(bytes)
      })), Stream.run(fs.sink(installer, { flag: "w" })))
      if (copied !== candidate.manifest.artifact.bytes || digest.digest("hex") !== candidate.manifest.artifact.sha256) {
        return yield* new ApplicationUpdateFailed({ message: "The downloaded installer failed publisher verification." })
      }
      yield* permissions.protectFile(installer)
      yield* verifier.verify(installer)
      const helper = win32.join(directory, "magnitude-update.exe")
      yield* permissions.createFile(helper)
      yield* fs.stream(options.cliPath).pipe(Stream.run(fs.sink(helper, { flag: "w" })))
      yield* permissions.protectFile(helper)
      yield* Ref.set(prepared, Option.some({ preparedDirectory: directory, stateDirectory: options.stateDirectory,
        applicationPath: options.applicationPath, version: candidate.manifest.version, envelope: candidate.envelope }))
    }).pipe(Effect.onError(() => fs.remove(directory, { recursive: true }).pipe(Effect.ignore)))
  }).pipe(Effect.mapError(() => new ApplicationUpdateFailed({ message: "Could not verify and prepare the Windows application installer." }))))
  const discard = Ref.getAndSet(prepared, Option.none()).pipe(Effect.flatMap(Option.match({
    onNone: () => Effect.void,
    onSome: request => fs.remove(request.preparedDirectory, { recursive: true }).pipe(Effect.ignore),
  })))
  return { source, previousFailure: prior, discard, restart: (showWindow: boolean) => Ref.get(prepared).pipe(Effect.flatMap(Option.match({
    onNone: () => new ApplicationUpdateFailed({ message: "Download the application update before restarting." }),
    onSome: request => startWindowsUpdateHandoff({ ...request, showWindow }).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message }))),
  }))) }
})
