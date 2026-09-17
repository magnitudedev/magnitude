import { FileSystem } from "@effect/platform"
import { Effect, Stream } from "effect"
import { randomUUID } from "node:crypto"
import { join } from "node:path"
import { PreparedUpdateStore, PrivateFilePermissions, MacUpdateHandoff } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateFailed } from "./application-update"
import { NativeMacUpdate, stageMacUpdateArchive } from "./mac-update-stage"
import { PreparedUpdateInstaller } from "./prepared-update-installation"
import { hostedUpdateSource, type HostedUpdateSourceOptions } from "./hosted-update-source"

export const macUpdateSource = (options: HostedUpdateSourceOptions & {
  readonly stateDirectory: string
  readonly bundle: string
  readonly cliPath: string
  readonly addonPath: string
}) => Effect.gen(function* () {
  const native = yield* NativeMacUpdate
  const relaunch = yield* MacUpdateHandoff
  const store = yield* PreparedUpdateStore
  const fs = yield* FileSystem.FileSystem
  const permissions = yield* PrivateFilePermissions
  const parent = join(options.stateDirectory, "update-helpers")
  if (yield* fs.exists(parent)) {
    for (const name of yield* fs.readDirectory(parent)) {
      if (/^helper-[a-f0-9-]{36}$/.test(name)) yield* fs.remove(join(parent, name), { recursive: true, force: true })
    }
  }
  return {
    source: hostedUpdateSource(options, (archive, release) => store.prepare(archive, release).pipe(
      Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })),
    )),
    installer: PreparedUpdateInstaller.of({
      requiresAuthorization: false,
      install: (archive, _release, showWindow) => Effect.scoped(Effect.gen(function* () {
        yield* permissions.prepareDirectory(parent)
        const helperDirectory = join(parent, `helper-${randomUUID()}`)
        yield* permissions.prepareDirectory(helperDirectory)
        yield* Effect.gen(function* () {
          for (const [source, name] of [[options.cliPath, "magnitude-update"], [options.addonPath, "desktop-host.node"]] as const) {
            const destination = join(helperDirectory, name)
            yield* permissions.createFile(destination)
            yield* fs.stream(source).pipe(Stream.run(fs.sink(destination, { flag: "r+" })))
            yield* fs.chmod(destination, name === "magnitude-update" ? 0o700 : 0o600)
          }
          const handoff = yield* relaunch.start({ helperDirectory, stateDirectory: options.stateDirectory, bundle: options.bundle, showWindow })
          yield* stageMacUpdateArchive(archive).pipe(Effect.provideService(NativeMacUpdate, native))
          yield* handoff.commit
        }).pipe(Effect.onError(() => fs.remove(helperDirectory, { recursive: true, force: true }).pipe(Effect.ignore)))
      })).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error instanceof Error ? error.message : "Could not prepare the application update." }))),
    }),
  }
})
