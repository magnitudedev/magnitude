import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import {
  GlobalStorage,
  CustomEndpointNameSchema,
  ProjectStorage,
  StorageLive,
  VersionLive,
  makeGlobalStorage,
  makeProjectStorage,
} from "@magnitudedev/storage"
import { CustomEndpoints, CustomEndpointsLive } from "./custom-endpoints"

const configuredEndpoint = (displayName: string) => ({
  providers: {
    openrouter: {
      displayName,
      connection: {
        baseUrl: "https://openrouter.ai/api/v1",
        authentication: { type: "none" },
      },
      models: {
        "z-ai/glm-5.2": {
          displayName: "GLM 5.2",
          contextWindow: 1048576,
          maxOutputTokens: 128000,
        },
      },
    },
  },
})

describe("custom endpoint configuration observation", () => {
  let root: string

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-custom-endpoints-"))
  })

  afterEach(async () => {
    await rm(root, { recursive: true, force: true })
  })

  it("retains the last good declarations when a later read is invalid", async () => {
    const globalStorage = makeGlobalStorage({ root })
    await Bun.write(globalStorage.paths.configFile, JSON.stringify(configuredEndpoint("OpenRouter")))

    const platform = Layer.mergeAll(
      BunFileSystem.layer,
      BunPath.layer,
      Layer.succeed(GlobalStorage, GlobalStorage.of(globalStorage)),
      Layer.succeed(ProjectStorage, ProjectStorage.of(makeProjectStorage({ cwd: root }))),
      VersionLive("test"),
    )
    const storage = StorageLive.pipe(Layer.provide(platform))
    const customEndpoints = CustomEndpointsLive.pipe(
      Layer.provide(Layer.merge(platform, storage)),
    )

    const observed = await Effect.runPromise(Effect.gen(function* () {
      const endpoints = yield* CustomEndpoints
      const initial = yield* endpoints.get

      yield* Effect.promise(() => Bun.write(globalStorage.paths.configFile, "{ partial"))
      yield* Effect.sleep("1200 millis")
      const afterMalformed = yield* endpoints.get

      yield* Effect.promise(() => Bun.write(
        globalStorage.paths.configFile,
        JSON.stringify(configuredEndpoint("OpenRouter updated")),
      ))
      yield* Effect.sleep("1200 millis")
      const afterValid = yield* endpoints.get

      return { initial, afterMalformed, afterValid }
    }).pipe(Effect.provide(customEndpoints)))

    const openrouter = CustomEndpointNameSchema.make("openrouter")
    expect(observed.initial[openrouter]?.displayName).toBe("OpenRouter")
    expect(observed.afterMalformed[openrouter]?.displayName).toBe("OpenRouter")
    expect(observed.afterValid[openrouter]?.displayName).toBe("OpenRouter updated")
  }, 10_000)
})
