import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { mkdtemp, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { makeGlobalStoragePaths, makeProjectStoragePaths } from "../paths"
import { GlobalStorage } from "../services"
import { ProjectStorage } from "../services/project-storage"
import { Version } from "../services/version"
import { MagnitudeStorage, StorageLive } from "../storage"

describe("new application-state initialization", () => {
  let root: string

  const dependencies = () => Layer.mergeAll(
    BunFileSystem.layer,
    BunPath.layer,
    Layer.succeed(Version, Version.of({ getVersion: () => "0.0.0-test" })),
    Layer.succeed(GlobalStorage, GlobalStorage.of({
      root,
      paths: makeGlobalStoragePaths(root),
    })),
    Layer.succeed(ProjectStorage, ProjectStorage.of({
      cwd: "/repo",
      root: join(root, "project"),
      paths: makeProjectStoragePaths(root),
    })),
  )

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-state-defaults-"))
  })

  afterEach(async () => {
    await rm(root, { recursive: true, force: true })
  })

  it("creates independent defaults without reading authored configuration", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      contextLimits: { softCapRatio: 0.5, softCapMaxTokens: null },
    }))

    const state = await Effect.runPromise(Effect.gen(function* () {
      const storage = yield* MagnitudeStorage
      return {
        models: yield* storage.models.get,
        onboarding: yield* storage.onboarding.get,
      }
    }).pipe(Effect.provide(StorageLive.pipe(Layer.provide(dependencies())))))

    expect(state.models).toEqual({
      configurations: [],
      slots: { primary: Option.none(), secondary: Option.none() },
      recentModels: { primary: [], secondary: [] },
      favorites: [],
      configurationRecoveryCompleted: false,
    })
    expect(state.onboarding).toEqual({ completed: false })
    expect(await Bun.file(paths.modelsFile).exists()).toBe(true)
    expect(await Bun.file(paths.onboardingFile).exists()).toBe(true)
  })
})
