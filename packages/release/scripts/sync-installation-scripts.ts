import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect } from "effect"
import { fileURLToPath } from "node:url"
import { syncInstallationScripts } from "./build/sync-installation-scripts"

const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const publicKey = yield* fs.readFileString(
    yield* Config.string("MAGNITUDE_INSTALL_PUBLIC_KEY").pipe(
      Config.withDefault(fileURLToPath(new URL("../resources/distribution/magnitude-2026-01.pub.pem", import.meta.url))),
    ),
  )
  const dryRun = yield* Config.boolean("MAGNITUDE_SYNC_DRY_RUN").pipe(Config.withDefault(false))
  const result = yield* syncInstallationScripts({
    landingRepo: yield* Config.string("MAGNITUDE_LANDING_REPO"),
    branch: yield* Config.string("MAGNITUDE_LANDING_BRANCH").pipe(Config.withDefault("main")),
    version: yield* Config.string("MAGNITUDE_RELEASE_VERSION"),
    sourceCommit: yield* Config.string("MAGNITUDE_SOURCE_COMMIT"),
    origin: yield* Config.string("MAGNITUDE_INSTALL_ORIGIN").pipe(Config.withDefault("https://magnitude.dev")),
    appleTeam: yield* Config.string("APPLE_TEAM_ID"),
    windowsPublisher: yield* Config.string("MAGNITUDE_WINDOWS_PUBLISHER"),
    publicKey,
    push: !dryRun,
  })
  yield* Effect.logInfo("Installation scripts sync finished", { ...result, dryRun })
})

BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
