import { FileSystem } from "@effect/platform"
import { Effect, Stream } from "effect"
import { randomUUID } from "node:crypto"
import { win32 } from "node:path"
import { PreparedUpdateStore, PrivateFilePermissions, WindowsInstallerVerifier, startWindowsUpdateHandoff } from "@magnitudedev/daemon-management/desktop-native"
import { ApplicationUpdateFailed } from "./application-update"
import { PreparedUpdateInstaller } from "./prepared-update-installation"
import { hostedUpdateSource, type HostedUpdateSourceOptions } from "./hosted-update-source"

export const makeWindowsUpdateSource = (options: HostedUpdateSourceOptions & {
  readonly stateDirectory: string
  readonly applicationPath: string
  readonly cliPath: string
  readonly addonPath: string
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const permissions = yield* PrivateFilePermissions
  const verifier = yield* WindowsInstallerVerifier
  const store = yield* PreparedUpdateStore
  const parent = win32.join(options.stateDirectory, "update-helpers")
  // Bootstrap holds application ownership and has excluded a live installer. A just-exited
  // Windows helper may still have its executable open; leave that directory for the next launch.
  if (yield* fs.exists(parent)) {
    for (const name of yield* fs.readDirectory(parent)) {
      if (/^helper-[a-f0-9-]{36}$/.test(name)) yield* fs.remove(win32.join(parent, name), { recursive: true, force: true }).pipe(Effect.ignore)
    }
  }
  return {
    source: yield* hostedUpdateSource(options, (archive, release) => verifier.verify(archive).pipe(
      Effect.zipRight(store.prepare(archive, release)),
      Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })),
    )),
    installer: PreparedUpdateInstaller.of({
      requiresAuthorization: false,
      install: (archive, release, continuation) => Effect.gen(function* () {
        yield* verifier.verify(archive)
        yield* permissions.prepareDirectory(parent)
        const helperDirectory = win32.join(parent, `helper-${randomUUID()}`)
        yield* permissions.prepareDirectory(helperDirectory)
        yield* Effect.gen(function* () {
          // Only the helper and native lock adapter must survive replacement of the installed app.
          for (const [source, name] of [[options.cliPath, "magnitude.exe"], [options.addonPath, "desktop-host.node"]] as const) {
            const destination = win32.join(helperDirectory, name)
            yield* permissions.createFile(destination)
            yield* fs.stream(source).pipe(Stream.run(fs.sink(destination, { flag: "r+" })))
            yield* permissions.protectFile(destination)
          }
          yield* startWindowsUpdateHandoff({ helperDirectory, dataDirectory: options.dataDirectory,
            stateDirectory: options.stateDirectory, applicationPath: options.applicationPath, release, continuation })
        }).pipe(Effect.onError(() => fs.remove(helperDirectory, { recursive: true, force: true }).pipe(Effect.ignore)))
      }).pipe(Effect.mapError(() => new ApplicationUpdateFailed({ message: "Could not verify and start the Windows application installer." }))),
    }),
  }
})
