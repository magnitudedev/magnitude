import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect } from "effect"
import { join } from "node:path"
import { buildWindowsDesktopInstaller } from "../build/desktop-windows"

/** Exercises the production installer with a small inert payload; this is not desktop runtime acceptance. */
const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* Config.string("MAGNITUDE_INSTALLER_TEST_ROOT")
  const makensis = yield* Config.string("MAGNITUDE_INSTALLER_TEST_NSIS")
  const app = join(root, "fixture-app")
  yield* fs.makeDirectory(join(app, "resources"), { recursive: true })
  for (const file of ["Magnitude.exe", "resources/magnitude-service.exe"]) {
    yield* fs.copyFile(join(root, "windows-installer-test.exe"), join(app, file))
  }
  yield* fs.copyFile(join(root, "MagnitudeInstallGuard.dll"), join(app, "resources/desktop-host.node"))
  yield* fs.writeFileString(join(app, "resources/app.asar"), "Installer fixture; not an Electron application")
  yield* fs.writeFileString(join(app, "resources/Magnitude-LICENSE.txt"), "Installer acceptance fixture")
  for (const version of ["1.2.3", "1.2.4"]) {
    yield* fs.writeFileString(join(app, "resources/fixture-version.txt"), version)
    yield* buildWindowsDesktopInstaller({ app, guard: join(root, "MagnitudeInstallGuard.dll"), makensis,
      version, revision: 0, output: join(root, version) })
  }
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
