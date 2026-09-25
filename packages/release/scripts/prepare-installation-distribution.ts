import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Schema } from "effect"
import { fileURLToPath } from "node:url"
import { PublishedUpdate } from "../src/hosted-update/manifest"
import { writeInstallationDistribution } from "./build/installation-distribution"

const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const publications = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Array(PublishedUpdate)))(
    yield* fs.readFileString(yield* Config.string("MAGNITUDE_INSTALL_PUBLICATIONS")))
  const publicKey = yield* fs.readFileString(yield* Config.string("MAGNITUDE_INSTALL_PUBLIC_KEY").pipe(
    Config.withDefault(fileURLToPath(new URL("../resources/distribution/magnitude-2026-01.pub.pem", import.meta.url)))))
  const result = yield* writeInstallationDistribution({
    output: yield* Config.string("MAGNITUDE_INSTALL_OUTPUT"),
    origin: yield* Config.string("MAGNITUDE_INSTALL_ORIGIN").pipe(Config.withDefault("https://magnitude.dev")),
    appleTeam: yield* Config.string("APPLE_TEAM_ID"),
    windowsPublisher: yield* Config.string("MAGNITUDE_WINDOWS_PUBLISHER"), publicKey, publications,
  })
  yield* Effect.logInfo("Installation distribution prepared", result)
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
