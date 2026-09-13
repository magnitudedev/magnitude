import { FileSystem } from "@effect/platform"
import { Effect, Option, Ref, Schema } from "effect"
import { dirname, join } from "node:path"
import { isNewerVersion } from "@magnitudedev/release"
import { LinuxPackageUpdate, LinuxUpdateResult, startLinuxUpdateHandoff } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateFailed } from "./application-update"
import { hostedUpdateSource, type HostedUpdateSourceOptions } from "./hosted-update-source"

const Prepared = Schema.Struct({ requestPath: Schema.String, version: Schema.String })
export const makeLinuxUpdateSource = (options: HostedUpdateSourceOptions & { readonly stateDirectory: string }) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const prepared = yield* Ref.make(Option.none<typeof Prepared.Type>())
  const resultPath = join(options.stateDirectory, "update-result.json")
  const previousFailure = (yield* fs.exists(resultPath)) ? yield* fs.readFileString(resultPath).pipe(
    Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(LinuxUpdateResult))),
    Effect.map(result => isNewerVersion(result.version, options.metadata.version) ? result.error : Option.none()),
    Effect.catchAll(() => Effect.succeed(Option.some("The previous application update result could not be read. Check for updates to retry."))),
  ) : Option.none<string>()
  const source = hostedUpdateSource(options, (archive, candidate) => Effect.gen(function* () {
    const parent = join(options.stateDirectory, "application-updates")
    yield* fs.makeDirectory(parent, { recursive: true, mode: 0o700 })
    const directory = yield* fs.makeTempDirectory({ directory: parent, prefix: "prepared-" })
    yield* fs.chmod(directory, 0o700)
    yield* Effect.gen(function* () {
      const packagePath = join(directory, `magnitude.${candidate.manifest.artifact.target.package}`)
      yield* fs.copyFile(archive, packagePath)
      yield* fs.chmod(packagePath, 0o600)
      const requestPath = join(directory, "request.json")
      yield* fs.writeFileString(requestPath, yield* Schema.encode(Schema.parseJson(LinuxPackageUpdate))({ packagePath, envelope: candidate.envelope }), { flag: "wx", mode: 0o600 })
      yield* Ref.set(prepared, Option.some({ requestPath, version: candidate.manifest.version }))
    }).pipe(Effect.onError(() => fs.remove(directory, { recursive: true }).pipe(Effect.ignore)))
  }).pipe(Effect.mapError(() => new ApplicationUpdateFailed({ message: "Could not prepare the application package for installation." }))))
  const discard = Ref.getAndSet(prepared, Option.none()).pipe(Effect.flatMap(Option.match({
    onNone: () => Effect.void,
    onSome: request => fs.remove(dirname(request.requestPath), { recursive: true }).pipe(Effect.ignore),
  })))
  return { source, previousFailure, discard, restart: (showWindow: boolean) => Ref.get(prepared).pipe(Effect.flatMap(Option.match({
    onNone: () => new ApplicationUpdateFailed({ message: "Download the application update before restarting." }),
    onSome: request => startLinuxUpdateHandoff({ ...request, stateDirectory: options.stateDirectory, showWindow }).pipe(
      Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message }))),
  }))) }
})
