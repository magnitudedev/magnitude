import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { dirname, isAbsolute, join, resolve, basename } from "node:path"
import { ApplicationUpdateFailed } from "./application-update"
import { completeMacPreparedInstallation } from "./mac-prepared-installation"
import { readInstalledUpdateConfiguration } from "./update-configuration"
import { makePreparedUpdateStore, PreparedUpdateStore } from "../desktop-native/prepared-update"
import { acquireApplicationMaintenance } from "../desktop-native/application-owner"
import { acquireUpdateInstallationLease } from "../desktop-native/update-installation-lease"
import { MacUpdateAdmission, MAC_UPDATE_LEASE_DESCRIPTOR, nativeMacUpdateAdmission } from "../desktop-native/mac-update-lease"
import { MacUpdateFilesystem, nativeMacUpdateFilesystem } from "../desktop-native/mac-update-filesystem"
import { nativeMacBundleVerifier } from "../desktop-native/mac-update-validation"
import { nativeHostLayer } from "../desktop-native/index"
import { unixPrivateFilePermissions } from "../desktop-native/private-files"
import { NativeMacApplicationInstallation } from "../desktop-native/mac-update-installation"
import { guardedCommandLayer } from "../desktop-native/guarded-command"
import { MacUpdateArchiveStager, makeMacUpdateArchiveStager } from "../desktop-native/mac-update-staging"
import { makeUnixProcessContinuation } from "../desktop-native/unix-continuation"

const Path = Schema.NonEmptyString.pipe(Schema.maxLength(4096), Schema.filter(path => isAbsolute(path) && resolve(path) === path && !path.includes("\0")))
export const MacInstallerRequest = Schema.Struct({
  protocol: Schema.Literal(1), bundle: Path.pipe(Schema.endsWith(".app")), stateDirectory: Path, dataDirectory: Path,
  continuation: Schema.Union(Schema.TaggedStruct("None", {}), Schema.TaggedStruct("Foreground", {
    arguments: Schema.Array(Schema.String.pipe(Schema.maxLength(4096), Schema.filter(value => !value.includes("\0")))).pipe(
      Schema.minItems(1), Schema.maxItems(4096), Schema.filter(args => args[0] === "serve")),
  })),
})
export type MacInstallerRequest = typeof MacInstallerRequest.Type
const failed = () => new ApplicationUpdateFailed({ message: "The application installer invocation is invalid or could not complete." })
export const decodeMacInstallerInvocation = (payload: string, executable: string, descriptor: string | undefined) => Effect.gen(function* () {
  if (Buffer.byteLength(payload) > 64 * 1024 || !descriptor || !/^[1-9][0-9]{0,9}$/.test(descriptor)) return yield* failed()
  const fd = Number(descriptor)
  if (fd < 3 || fd > 2147483647) return yield* failed()
  const request = yield* Schema.decodeUnknown(Schema.parseJson(MacInstallerRequest))(payload, { onExcessProperty: "error" })
  const directory = dirname(executable)
  if (resolve(executable) !== executable || basename(executable) !== "magnitude" ||
      dirname(directory) !== join(request.stateDirectory, "mac-installers") || !/^installer-[A-Za-z0-9]+$/.test(basename(directory))) return yield* failed()
  return { request, directory, descriptor: fd }
}).pipe(Effect.mapError(failed))

/** Runs from the verified private copy, consumes inherited exclusion, and never starts a background server. */
export const runMacInstallerCommand = (payload: string, version: string) => Effect.gen(function* () {
  if (process.platform !== "darwin") return yield* failed()
  const invocation = yield* decodeMacInstallerInvocation(payload, process.execPath, process.env[MAC_UPDATE_LEASE_DESCRIPTOR])
  yield* Effect.sync(() => { delete process.env[MAC_UPDATE_LEASE_DESCRIPTOR] })
  const { request, directory, descriptor } = invocation
  const addon = join(directory, "desktop-host.node")
  const continuation = yield* makeUnixProcessContinuation(addon)
  const architecture = yield* Schema.decodeUnknown(Schema.Literal("arm64", "x64"))(process.arch)
  const result = yield* Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const native = yield* MacUpdateFilesystem
    const helperParent = yield* native.open(dirname(directory), true)
    const helper = yield* native.open(directory, true)
    if (helper.path !== (yield* fs.realPath(directory))) return yield* failed()
    const admission = yield* MacUpdateAdmission
    const lease = yield* admission.adopt(request.bundle, descriptor)
    yield* acquireApplicationMaintenance(request.stateDirectory)
    yield* acquireUpdateInstallationLease(request.stateDirectory)
    yield* Effect.addFinalizer(() => native.removeTree(helperParent, basename(directory), helper.identity).pipe(Effect.ignore))
    const configuration = yield* readInstalledUpdateConfiguration(directory)
    const store = yield* makePreparedUpdateStore({ dataDirectory: request.dataDirectory,
      target: { os: "darwin", arch: architecture, package: "mac-zip" }, trustedPublishers: configuration.trustedPublishers })
    const stager = yield* makeMacUpdateArchiveStager({ helper: join(directory, "magnitude-extract"), architecture,
      trustedPublishers: configuration.trustedPublishers })
    return yield* completeMacPreparedInstallation({ bundle: request.bundle, version, architecture }, lease).pipe(
      Effect.provideService(PreparedUpdateStore, store), Effect.provideService(MacUpdateArchiveStager, stager))
  }).pipe(Effect.provide([nativeHostLayer(addon), nativeMacUpdateAdmission(addon), nativeMacUpdateFilesystem(addon),
    nativeMacBundleVerifier(addon), NativeMacApplicationInstallation, guardedCommandLayer(join(directory, "magnitude-command")), unixPrivateFilePermissions])))
  if (result._tag !== "Installed") return yield* new ApplicationUpdateFailed({ message: "The previous installation was preserved. Retry the update explicitly." })
  if (request.continuation._tag === "Foreground") return yield* continuation.replace(join(request.bundle, "Contents/Resources/magnitude"), request.continuation.arguments, process.env)
  yield* Effect.sync(() => { process.stdout.write("The Magnitude update was installed.\n") })
}).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
