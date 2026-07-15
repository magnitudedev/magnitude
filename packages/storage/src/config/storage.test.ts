import { afterEach, beforeEach, describe, expect, test } from "vitest"
import { mkdtemp, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { makeGlobalStoragePaths } from "../paths"
import { GlobalStorage } from "../services"
import { makeConfigStorage } from "./storage"

describe("config storage onboarding state", () => {
  let root: string

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-onboarding-config-"))
  })

  afterEach(async () => {
    await rm(root, { recursive: true, force: true })
  })

  const run = <A, E>(effect: Effect.Effect<A, E, never>) => Effect.runPromise(effect)

  test("persists completion atomically without replacing model slots", async () => {
    const base = Layer.mergeAll(
      BunFileSystem.layer,
      BunPath.layer,
      Layer.succeed(GlobalStorage, GlobalStorage.of({ root, paths: makeGlobalStoragePaths(root) })),
    )
    const result = await run(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      yield* config.updateModelConfig({
        primary: { providerId: "llamacpp", providerModelId: "model" },
        secondary: { providerId: "llamacpp", providerModelId: "model" },
      })
      yield* config.completeCliModelSetupOnboarding(2, "2026-07-14T22:00:00.000Z")
      return yield* config.load()
    }).pipe(Effect.provide(base)))

    expect(result.models?.slots?.primary).toEqual({ providerId: "llamacpp", providerModelId: "model" })
    expect(result.models?.slots?.secondary).toEqual({ providerId: "llamacpp", providerModelId: "model" })
    expect(result.onboarding).toEqual({
      cliModelSetupVersion: 2,
      completedAt: "2026-07-14T22:00:00.000Z",
    })
  })
})
