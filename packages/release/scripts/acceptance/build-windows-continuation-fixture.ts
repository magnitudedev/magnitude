import { Command, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect } from "effect"
import { join } from "node:path"
import { buildWindowsDesktopInstaller } from "../build/desktop-windows"

/** The production installer with a mapped compiled CLI and addon; no inference payload. */
const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* Config.string("MAGNITUDE_INSTALLER_TEST_ROOT")
  const makensis = yield* Config.string("MAGNITUDE_INSTALLER_TEST_NSIS")
  const app = join(root, "foreground-app")
  yield* fs.makeDirectory(join(app, "resources"), { recursive: true })
  const code = yield* Command.exitCode(Command.make(process.execPath, "build", join(import.meta.dirname, "windows-foreground-probe.ts"),
    "--compile", "--outfile", join(app, "resources/magnitude.exe")))
  if (code !== 0) return yield* Effect.dieMessage("Foreground probe compilation failed")
  for (const file of ["Magnitude.exe", "resources/magnitude-service.exe"]) {
    yield* fs.copyFile(join(root, "windows-installer-test.exe"), join(app, file))
  }
  yield* fs.copyFile(join(root, "desktop-host.node"), join(app, "resources/desktop-host.node"))
  yield* fs.writeFileString(join(app, "resources/app.asar"), "Native continuation fixture")
  yield* fs.writeFileString(join(app, "resources/Magnitude-LICENSE.txt"), "Installer acceptance fixture")
  for (const version of ["1.2.3", "1.2.4", "1.2.5"]) {
    yield* fs.writeFileString(join(app, "resources/fixture-version.txt"), version)
    yield* buildWindowsDesktopInstaller({ app, guard: join(root, "MagnitudeInstallGuard.dll"), makensis,
      version, revision: 0, output: join(root, version) })
  }
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
