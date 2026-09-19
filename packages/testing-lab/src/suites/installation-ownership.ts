import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { join, relative, isAbsolute } from "node:path"
import { ApplicationIdentity } from "../application-identity"
import type { DesktopDriver } from "../desktop-driver"
import { AssertionFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { CliTests } from "./cli"

export const InstallationOwnership = Schema.Struct({ cli: Schema.String, bundledCli: Schema.String,
  before: ApplicationIdentity, after: ApplicationIdentity })

/** The installer-selected entry point must resolve to this package, even when it is a system symlink. */
export const verifyInstalledCliPath = (app: InstalledApplication) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const os = app.candidate.target.os
  const root = yield* fs.realPath(app.root)
  const bundledCli = yield* fs.realPath(join(app.root, os === "macos" ? "Contents/Resources" : "resources", os === "windows" ? "magnitude.exe" : "magnitude"))
  const within = relative(root, bundledCli)
  if (!within || within === ".." || within.startsWith("../") || within.startsWith("..\\") || isAbsolute(within)
    || (yield* fs.stat(bundledCli)).type !== "File" || (yield* fs.realPath(app.cli)) !== bundledCli) {
    return yield* new AssertionFailure({ message: "Installed CLI entry point does not resolve to the owned bundled executable" })
  }
  return bundledCli
})

export const verifyInstallationOwnership = (app: InstalledApplication, driver: Pick<DesktopDriver, "ready" | "identity">) => Effect.gen(function* () {
  const bundledCli = yield* verifyInstalledCliPath(app)
  const cli = yield* CliTests
  yield* driver.ready()
  const before = yield* driver.identity()
  yield* cli.version
  yield* cli.ensureService
  const after = yield* driver.identity()
  if (!Schema.equivalence(ApplicationIdentity)(before, after)) return yield* new AssertionFailure({ message: "Installed CLI changed the owning application or service identity" })
  return InstallationOwnership.make({ cli: app.cli, bundledCli, before, after })
})
