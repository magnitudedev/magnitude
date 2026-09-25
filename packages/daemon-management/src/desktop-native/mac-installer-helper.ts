import { FileSystem } from "@effect/platform"
import { APPLE_TEAM_ID } from "@magnitudedev/release/trust"
import { Context, Effect, Layer, Schema, Stream } from "effect"
import { dirname, join } from "node:path"
import { MacBundleVerifier } from "./mac-update-validation"
import { MacUpdateFilesystem } from "./mac-update-filesystem"
import { GuardedCommand } from "./guarded-command"
import type { MacExclusiveInstallationLease } from "./mac-update-lease"
import { readInstalledUpdateConfiguration, ApplicationUpdateConfiguration } from "../application-update/update-configuration"

export class MacInstallerHelperFailed extends Schema.TaggedError<MacInstallerHelperFailed>()("MacInstallerHelperFailed", {}) {
  override get message() { return "The application installer helper could not be prepared or verified." }
}
export interface MacInstallerCodeVerifier {
  readonly verify: (path: string) => Effect.Effect<void, MacInstallerHelperFailed>
}
export const MacInstallerCodeVerifier = Context.GenericTag<MacInstallerCodeVerifier>("@magnitudedev/daemon-management/MacInstallerCodeVerifier")
export const nativeMacInstallerCodeVerifier = Layer.effect(MacInstallerCodeVerifier, Effect.gen(function* () {
  if (!/^[A-Z0-9]{10}$/.test(APPLE_TEAM_ID)) return yield* new MacInstallerHelperFailed()
  const runner = yield* GuardedCommand
  const requirement = `anchor apple generic and certificate leaf[subject.OU] = "${APPLE_TEAM_ID}" and certificate leaf[field.1.2.840.113635.100.6.1.13] exists`
  return MacInstallerCodeVerifier.of({ verify: path => runner.run("/usr/bin/codesign",
    ["--verify", "--strict", "--all-architectures", "-R", `=${requirement}`, path], {}).pipe(
    Effect.timeout("30 seconds"), Effect.filterOrFail(result => result.code === 0, () => new MacInstallerHelperFailed()),
    Effect.asVoid, Effect.mapError(() => new MacInstallerHelperFailed())) })
}))

/** Scoped until exec. The copied helper owns cleanup after it adopts installation admission. */
export const prepareMacInstallerHelper = (options: {
  readonly resources: string
  readonly stateDirectory: string
  readonly version: string
  readonly architecture: "arm64" | "x64"
  readonly lease: MacExclusiveInstallationLease
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const native = yield* MacUpdateFilesystem
  const verifier = yield* MacBundleVerifier
  const code = yield* MacInstallerCodeVerifier
  const bundle = dirname(dirname(options.resources))
  if (options.lease.bundle !== bundle) return yield* new MacInstallerHelperFailed()
  yield* options.lease.validate
  yield* verifier.verify(bundle, { version: options.version, architecture: options.architecture })
  const configuration = yield* readInstalledUpdateConfiguration(options.resources)
  const parent = join(options.stateDirectory, "mac-installers")
  yield* fs.makeDirectory(parent, { mode: 0o700 }).pipe(Effect.catchAll(error =>
    error._tag === "SystemError" && error.reason === "AlreadyExists" ? Effect.void : Effect.fail(error)))
  yield* native.open(parent, true)
  const directory = yield* fs.makeTempDirectoryScoped({ directory: parent, prefix: "installer-" })
  const retained = yield* native.open(directory, true)
  for (const name of ["magnitude", "desktop-host.node", "magnitude-command", "magnitude-extract"]) {
    const source = join(options.resources, name), destination = join(directory, name)
    const canonical = join(yield* fs.realPath(options.resources), name)
    if ((yield* fs.realPath(source)) !== canonical || (yield* fs.stat(source)).type !== "File") return yield* new MacInstallerHelperFailed()
    yield* fs.stream(source).pipe(Stream.run(fs.sink(destination, { flag: "wx", mode: 0o600 })))
    yield* fs.chmod(destination, name.endsWith(".node") ? 0o600 : 0o700)
    yield* code.verify(destination)
    yield* Effect.scoped(fs.open(destination, { flag: "r" }).pipe(Effect.flatMap(file => file.sync)))
  }
  const encoded = yield* Schema.encode(Schema.parseJson(ApplicationUpdateConfiguration))(configuration)
  const configurationPath = join(directory, "update-configuration.json")
  yield* fs.writeFileString(configurationPath, encoded, { flag: "wx", mode: 0o600 })
  yield* Effect.scoped(fs.open(configurationPath, { flag: "r" }).pipe(Effect.flatMap(file => file.sync)))
  yield* native.sync(retained)
  yield* options.lease.validate
  return { directory: retained.path, stateDirectory: dirname(dirname(retained.path)),
    executable: join(retained.path, "magnitude"), addonPath: join(retained.path, "desktop-host.node") }
}).pipe(Effect.mapError(() => new MacInstallerHelperFailed()))
