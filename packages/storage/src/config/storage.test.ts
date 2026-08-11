import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { mkdtemp, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { makeGlobalStoragePaths } from "../paths"
import { GlobalStorage } from "../services"
import { makeConfigStorage } from "./storage"

describe("authored configuration boundary", () => {
  let root: string

  const dependencies = () => Layer.mergeAll(
    BunFileSystem.layer,
    BunPath.layer,
    Layer.succeed(GlobalStorage, GlobalStorage.of({
      root,
      paths: makeGlobalStoragePaths(root),
    })),
  )

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-config-"))
  })

  afterEach(async () => {
    await rm(root, { recursive: true, force: true })
  })

  it("decodes current authored configuration independently of unknown fields", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      unknownDomain: { value: 42 },
      contextLimits: { softCapRatio: 0.8, softCapMaxTokens: null },
    }))

    const loaded = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      return yield* config.load()
    }).pipe(Effect.provide(dependencies())))

    expect(loaded.contextLimits).toEqual({ softCapRatio: 0.8, softCapMaxTokens: null })
    expect(loaded.providers).toEqual(Option.none())
  })

  it("preserves unrelated unknown authored fields during a current-field update", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      futureDomain: { value: 42 },
    }))

    await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      yield* config.setContextLimitPolicy({ softCapRatio: 0.8, softCapMaxTokens: null })
    }).pipe(Effect.provide(dependencies())))

    const persisted = await Bun.file(paths.configFile).json()
    expect(persisted.futureDomain).toEqual({ value: 42 })
    expect(persisted.contextLimits).toEqual({ softCapRatio: 0.8, softCapMaxTokens: null })
  })
})
