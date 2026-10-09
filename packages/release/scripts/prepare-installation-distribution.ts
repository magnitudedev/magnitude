import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect } from "effect"
import { fileURLToPath } from "node:url"
import { writeInstallationDistribution } from "./build/installation-distribution"

const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const publicKey = yield* fs.readFileString(yield* Config.string("MAGNITUDE_INSTALL_PUBLIC_KEY").pipe(
    Config.withDefault(fileURLToPath(new URL("../resources/distribution/magnitude-2026-01.pub.pem", import.meta.url)))))
  const output = yield* Config.string("MAGNITUDE_INSTALL_OUTPUT")
  yield* writeInstallationDistribution({
    output,
    origin: yield* Config.string("MAGNITUDE_INSTALL_ORIGIN").pipe(Config.withDefault("https://magnitude.dev")),
    appleTeam: yield* Config.string("APPLE_TEAM_ID"),
    windowsPublisher: yield* Config.string("MAGNITUDE_WINDOWS_PUBLISHER"), publicKey,
  })
  yield* Effect.logInfo("Installation scripts prepared", { output })
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
