import type { UpdateRelease } from "@magnitudedev/release/hosted-update"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { ApplicationUpdateFailed } from "./application-update"
import { GuardedCommand } from "../desktop-native/guarded-command"
import { MacApplicationInstallation } from "../desktop-native/mac-update-installation"
import { MacUpdateFilesystem } from "../desktop-native/mac-update-filesystem"
import { MacUpdateArchiveStager } from "../desktop-native/mac-update-staging"
import { openMacUpdateWorkspace, MacUpdateInstallationBusy } from "../desktop-native/mac-update-workspace"

/** The verified bootstrap executes outside the destination and never starts an application owner. */
export const installMacApplicationArchive = (options: {
  readonly bundle: string
  readonly archive: string
  readonly release: UpdateRelease
  readonly architecture: "arm64" | "x64"
}) => Effect.scoped(Effect.gen(function* () {
  const workspace = Option.getOrThrow(yield* openMacUpdateWorkspace(options.bundle, true))
  const fs = yield* MacUpdateFilesystem
  const installation = yield* MacApplicationInstallation
  const existing = yield* fs.inspect(workspace.installed, workspace.installedName)
  if (Option.isSome(existing) && (yield* installation.isInstalling(options.bundle))) {
    return yield* new MacUpdateInstallationBusy()
  }
  yield* workspace.recover
  yield* workspace.retire
  const current = yield* fs.inspect(workspace.installed, workspace.installedName)
  const previous = yield* Option.match(current, {
    onNone: () => Effect.succeed(Option.none<string>()),
    onSome: () => Effect.gen(function* () {
      const runner = yield* GuardedCommand
      const result = yield* runner.run("/usr/bin/plutil", ["-extract", "CFBundleShortVersionString", "raw", "-o", "-", "--",
        join(options.bundle, "Contents/Info.plist")], {})
      if (result.code !== 0) return yield* new ApplicationUpdateFailed({ message: "The installed application version could not be read." })
      return Option.some(yield* Schema.decodeUnknown(Schema.NonEmptyString.pipe(Schema.maxLength(96)))(result.stdout.trim()))
    }),
  })
  yield* workspace.clearUnpublishedStaging
  const stager = yield* MacUpdateArchiveStager
  yield* stager.stage(options.archive, workspace.staging, options.release)
  const result = yield* Option.match(previous, {
    onNone: () => workspace.publish({ version: options.release.version, architecture: options.architecture }),
    onSome: version => workspace.exchange({ previous: version, replacement: options.release.version, architecture: options.architecture }),
  })
  if (result._tag !== "Installed") return yield* new ApplicationUpdateFailed({ message: "The previous application was preserved. Retry installation explicitly." })
  yield* workspace.retire
  return result
})).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message })))
