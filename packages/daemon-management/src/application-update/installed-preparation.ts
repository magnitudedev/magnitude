import { Effect, Option, Schema } from "effect"
import { ReleaseTarget, UpdateClientMetadata } from "@magnitudedev/release/hosted-update"
import { makePreparedUpdateStore, PreparedUpdateStore } from "../desktop-native/prepared-update"
import { makeUpdatePreferences } from "../desktop-native/update-preferences"
import { nativeWindowsInstallerVerifier, WindowsInstallerVerifier } from "../desktop-native/windows-update-signature"
import { ApplicationUpdateFailed } from "./application-update"
import { hostedUpdateSource } from "./hosted-update-source"
import { makeUpdateIdentity } from "./update-identity"
import { readLinuxUpdateMetadata } from "./update-metadata"
import { readInstalledUpdateConfiguration } from "./update-configuration"

/** Construction is observational. Source acquisition belongs after application or maintenance admission. */
export const makeInstalledUpdatePreparation = (options: {
  readonly resources: string
  readonly addonPath: string
  readonly dataDirectory: string
  readonly version: string
  readonly osVersion: string
  readonly platform: "darwin" | "linux" | "win32"
  readonly architecture: "arm64" | "x64"
  readonly isolated: boolean
}) => Effect.gen(function* () {
  const configuration = yield* readInstalledUpdateConfiguration(options.resources)
  const metadata = options.platform === "linux"
    ? yield* readLinuxUpdateMetadata(options.resources, options.version, options.osVersion)
    : yield* Schema.decodeUnknown(UpdateClientMetadata)({ version: options.version, os_version: options.osVersion,
      os: options.platform === "win32" ? "windows" : "darwin", arch: options.architecture,
      package: options.platform === "win32" ? "windows-exe" : "mac-zip" })
  const target = yield* Schema.decodeUnknown(ReleaseTarget)({ os: metadata.os, arch: metadata.arch, package: metadata.package })
  const store = yield* makePreparedUpdateStore({ dataDirectory: options.dataDirectory, target, trustedPublishers: configuration.trustedPublishers })
  const preferences = yield* makeUpdatePreferences(options.dataDirectory)
  const makeSource = Effect.gen(function* () {
    if (options.isolated && !configuration.acceptance) return yield* new ApplicationUpdateFailed({
      message: "Application updates are unavailable in this isolated build.",
    })
    const identity = yield* makeUpdateIdentity(options.dataDirectory)
    const stage = (archive: string, release: Parameters<typeof store.prepare>[1]) => Effect.gen(function* () {
      if (options.platform === "win32") {
        if (Option.isNone(configuration.windowsPublisher)) return yield* new ApplicationUpdateFailed({ message: "The Windows update publisher is missing." })
        yield* Effect.flatMap(WindowsInstallerVerifier, verifier => verifier.verify(archive)).pipe(
          Effect.provide(nativeWindowsInstallerVerifier(options.addonPath, configuration.windowsPublisher.value)))
      }
      yield* store.prepare(archive, release)
    }).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
    return yield* hostedUpdateSource({ ...configuration, metadata, sign: identity.sign,
      dataDirectory: options.dataDirectory, userAgent: `Magnitude/${options.version} ${options.architecture} ${metadata.os}/${options.osVersion}` }, stage).pipe(
        Effect.provideService(PreparedUpdateStore, store))
  }).pipe(Effect.mapError(() => new ApplicationUpdateFailed({ message: "Application update preparation could not be initialized." })))
  return { configuration, store, preferences, makeSource }
}).pipe(Effect.mapError(() => new ApplicationUpdateFailed({ message: "The installed application update configuration could not be initialized." })))
