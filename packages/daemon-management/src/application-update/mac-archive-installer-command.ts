import { FileSystem } from "@effect/platform"
import { InstallationOffer, InstallationChannel, verifyInstallationOffer } from "@magnitudedev/release/hosted-update"
import { Effect, Schema } from "effect"
import { homedir } from "node:os"
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path"
import { ApplicationUpdateFailed } from "./application-update"
import { installMacApplicationArchive } from "./mac-archive-installation"
import { readInstalledUpdateConfiguration } from "./update-configuration"
import { acquireApplicationMaintenance } from "../desktop-native/application-owner"
import { nativeHostLayer } from "../desktop-native/index"
import { nativeMacUpdateAdmission } from "../desktop-native/mac-update-lease"
import { nativeMacUpdateFilesystem } from "../desktop-native/mac-update-filesystem"
import { MacBundleVerifier, nativeMacBundleVerifier } from "../desktop-native/mac-update-validation"
import { NativeMacApplicationInstallation } from "../desktop-native/mac-update-installation"
import { guardedCommandLayer } from "../desktop-native/guarded-command"
import { MacUpdateArchiveStager, makeMacUpdateArchiveStager } from "../desktop-native/mac-update-staging"
import { makeMacCliRegistration } from "../desktop-native/mac-cli-registration"

const Path = Schema.NonEmptyString.pipe(Schema.maxLength(4096), Schema.filter(path => isAbsolute(path) && resolve(path) === path && !path.includes("\0")))
export const MacArchiveInstallerRequest = Schema.Struct({
  bundle: Path.pipe(Schema.endsWith(".app")), archive: Path, offer: InstallationOffer, channel: InstallationChannel,
})
export const decodeMacArchiveInstallerRequest = (payload: string) => Effect.gen(function* () {
  if (Buffer.byteLength(payload) > 16384) return yield* new ApplicationUpdateFailed({ message: "The application installation request is too large." })
  return yield* Schema.decodeUnknown(Schema.parseJson(MacArchiveInstallerRequest))(payload, { onExcessProperty: "error" })
})
const contains = (parent: string, child: string) => {
  const path = relative(parent, child)
  return path !== ".." && !path.startsWith("../") && !isAbsolute(path)
}

/** The shell verifies the extracted bootstrap bundle before invoking this finite entry point. */
export const runMacArchiveInstallerCommand = (payload: string, stateDirectory: string, version: string) => Effect.gen(function* () {
  if (process.platform !== "darwin") return yield* new ApplicationUpdateFailed({ message: "This installer requires macOS." })
  const request = yield* decodeMacArchiveInstallerRequest(payload)
  const fs = yield* FileSystem.FileSystem
  const executable = yield* fs.realPath(process.execPath)
  const resources = dirname(executable)
  const source = dirname(dirname(resources))
  const destinationParent = yield* fs.realPath(dirname(request.bundle))
  const bundle = join(destinationParent, basename(request.bundle))
  if (basename(executable) !== "magnitude" || basename(resources) !== "Resources" ||
      basename(dirname(resources)) !== "Contents" || !source.endsWith(".app") ||
      contains(bundle, source) || contains(source, bundle)) {
    return yield* new ApplicationUpdateFailed({ message: "Run the installer from the downloaded application outside the installation destination." })
  }
  const architecture = yield* Schema.decodeUnknown(Schema.Literal("arm64", "x64"))(process.arch)
  const addon = join(resources, "desktop-host.node")
  yield* Effect.scoped(Effect.gen(function* () {
    const verifier = yield* MacBundleVerifier
    yield* verifier.verify(source, { version, architecture })
    const configuration = yield* readInstalledUpdateConfiguration(resources)
    const offer = yield* verifyInstallationOffer(request.offer, { os: "darwin", arch: architecture, package: "mac-zip" },
      request.channel, configuration.trustedPublishers)
    yield* acquireApplicationMaintenance(stateDirectory)
    const stager = yield* makeMacUpdateArchiveStager({ helper: join(resources, "magnitude-extract"), architecture,
      trustedPublishers: configuration.trustedPublishers })
    yield* installMacApplicationArchive({ bundle, archive: request.archive, release: offer.release, architecture }).pipe(Effect.provideService(MacUpdateArchiveStager, stager))
    const environment = Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined))
    const registration = yield* makeMacCliRegistration({ home: homedir(), resourcesDirectory: join(bundle, "Contents/Resources"), environment })
    yield* registration.install
  }).pipe(Effect.provide([nativeHostLayer(addon), nativeMacUpdateAdmission(addon), nativeMacUpdateFilesystem(addon),
    nativeMacBundleVerifier(addon), NativeMacApplicationInstallation, guardedCommandLayer(join(resources, "magnitude-command"))])))
  yield* Effect.sync(() => { process.stdout.write("Magnitude was installed. Run magnitude serve to start the server.\n") })
}).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
