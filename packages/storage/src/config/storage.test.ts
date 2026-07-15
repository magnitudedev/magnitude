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

  test("persists a versioned completion atomically without replacing other domains", async () => {
    const base = Layer.mergeAll(
      BunFileSystem.layer,
      BunPath.layer,
      Layer.succeed(GlobalStorage, GlobalStorage.of({ root, paths: makeGlobalStoragePaths(root) })),
    )
    const result = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      yield* config.update((current) => ({
        ...current,
        models: {
          slots: {
            primary: { providerId: "llamacpp", providerModelId: "model" },
            secondary: { providerId: "llamacpp", providerModelId: "model" },
          },
        },
        localInference: {
          usage: { localModelRole: "subagent", sessionConcurrency: "up_to_three" },
        },
      }))
      yield* config.completeOnboardingFlow("model_setup", 2, "2026-07-14T22:00:00.000Z")
      return yield* config.load()
    }).pipe(Effect.provide(base)))

    expect(result.models?.slots?.primary).toEqual({ providerId: "llamacpp", providerModelId: "model" })
    expect(result.localInference?.usage).toEqual({
      localModelRole: "subagent",
      sessionConcurrency: "up_to_three",
    })
    expect(result.onboarding?.completions?.model_setup).toEqual({
      version: 2,
      completedAt: "2026-07-14T22:00:00.000Z",
    })
  })
})
