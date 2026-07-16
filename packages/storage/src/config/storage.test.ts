import { afterEach, beforeEach, describe, expect, test } from "vitest"
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { makeGlobalStoragePaths } from "../paths"
import { GlobalStorage } from "../services"
import { makeConfigStorage } from "./storage"

describe("config storage onboarding state", () => {
  let root: string

  const makeBase = () => Layer.mergeAll(
    BunFileSystem.layer,
    BunPath.layer,
    Layer.succeed(GlobalStorage, GlobalStorage.of({ root, paths: makeGlobalStoragePaths(root) })),
  )

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-onboarding-config-"))
  })

  afterEach(async () => {
    await rm(root, { recursive: true, force: true })
  })

  test("persists a versioned completion atomically without replacing other domains", async () => {
    const base = makeBase()
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

  test("recovers stale onboarding without replacing valid sibling domains", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      models: {
        slots: {
          primary: { providerId: "llamacpp", providerModelId: "model" },
        },
      },
      onboarding: {
        completions: {
          model_setup: { completedAt: "2026-07-14T22:00:00.000Z" },
        },
      },
      localInference: {
        usage: { localModelRole: "main", sessionConcurrency: "one" },
      },
      futureDomain: { enabled: true },
    }))

    const result = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      const onboarding = yield* config.getOnboardingConfig()
      const loaded = yield* config.load()
      return { onboarding, loaded }
    }).pipe(Effect.provide(makeBase())))

    expect(result.onboarding?.completions?.model_setup).toBeUndefined()
    expect(result.loaded.models?.slots?.primary).toEqual({
      providerId: "llamacpp",
      providerModelId: "model",
    })
    expect(result.loaded.localInference?.usage).toEqual({
      localModelRole: "main",
      sessionConcurrency: "one",
    })

    const persisted = await Bun.file(paths.configFile).json()
    expect(persisted.onboarding?.completions?.model_setup).toBeUndefined()
    expect(persisted.futureDomain).toEqual({ enabled: true })
  })

  test("resets an invalid binding while preserving local usage", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      localInference: {
        usage: { localModelRole: "subagent", sessionConcurrency: "up_to_three" },
        binding: {
          _tag: "Managed",
          selectionId: "selection",
          artifactId: "artifact",
          providerModelId: "provider-model",
          contextTokens: -1,
          parallelSlots: 3,
        },
      },
    }))

    const localInference = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      return yield* config.getLocalInferenceConfig()
    }).pipe(Effect.provide(makeBase())))

    expect(localInference?.usage).toEqual({
      localModelRole: "subagent",
      sessionConcurrency: "up_to_three",
    })
    expect(localInference?.binding).toBeUndefined()
    expect((await Bun.file(paths.configFile).json()).localInference.binding).toBeUndefined()
  })

  test("recovers an invalid model-slot leaf without replacing sibling slots", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      models: {
        slots: {
          primary: { providerId: 42, providerModelId: "primary-model" },
          secondary: { providerId: "cloud", providerModelId: "secondary-model" },
        },
      },
    }))

    const models = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      return yield* config.getModelConfig()
    }).pipe(Effect.provide(makeBase())))

    expect(models?.slots?.primary).toEqual({ providerModelId: "primary-model" })
    expect(models?.slots?.secondary).toEqual({
      providerId: "cloud",
      providerModelId: "secondary-model",
    })
  })

  test("can complete onboarding after recovering stale state", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      onboarding: { completions: { model_setup: { completedAt: "old" } } },
      models: { localSlotIntent: { primary: "local" } },
    }))

    const loaded = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      yield* config.completeOnboardingFlow("model_setup", 3, "2026-07-15T23:00:00.000Z")
      return yield* config.load()
    }).pipe(Effect.provide(makeBase())))

    expect(loaded.onboarding?.completions?.model_setup).toEqual({
      version: 3,
      completedAt: "2026-07-15T23:00:00.000Z",
    })
    expect(loaded.models?.localSlotIntent?.primary).toBe("local")
  })

  test("backs up malformed JSON and resets to the root default", async () => {
    const paths = makeGlobalStoragePaths(root)
    const original = "{ malformed"
    await Bun.write(paths.configFile, original)

    const loaded = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      return yield* config.load()
    }).pipe(Effect.provide(makeBase())))

    expect(loaded).toEqual({})
    expect(await Bun.file(paths.configFile).json()).toEqual({})
    const backup = (await readdir(root)).find((name) => name.startsWith("config.json.corrupt-"))
    expect(backup).toBeDefined()
    expect(await readFile(join(root, backup!), "utf8")).toBe(original)
  })

  test("backs up a structurally invalid root before resetting it", async () => {
    const paths = makeGlobalStoragePaths(root)
    const original = "42"
    await Bun.write(paths.configFile, original)

    const loaded = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      return yield* config.load()
    }).pipe(Effect.provide(makeBase())))

    expect(loaded).toEqual({})
    const backup = (await readdir(root)).find((name) => name.startsWith("config.json.corrupt-"))
    expect(backup).toBeDefined()
    expect(await readFile(join(root, backup!), "utf8")).toBe(original)
  })

  test("preserves unknown fields through a later config update", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      futureDomain: { value: 42 },
      models: { slots: {} },
    }))

    await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      yield* config.updateModelConfig({
        primary: { providerId: "llamacpp", providerModelId: "model" },
      })
    }).pipe(Effect.provide(makeBase())))

    const persisted = await Bun.file(paths.configFile).json()
    expect(persisted.futureDomain).toEqual({ value: 42 })
    expect(persisted.models.slots.primary.providerModelId).toBe("model")
  })

  test("serializes concurrent recovery-capable config updates", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      yield* config.save({ contextLimits: { softCapRatio: 0, softCapMaxTokens: null } })
      yield* Effect.all(
        Array.from({ length: 20 }, () => config.update((current) => ({
          ...current,
          contextLimits: {
            ...current.contextLimits,
            softCapRatio: (current.contextLimits?.softCapRatio ?? 0) + 1,
            softCapMaxTokens: current.contextLimits?.softCapMaxTokens ?? null,
          },
        }))),
        { concurrency: "unbounded" },
      )
      return yield* config.load()
    }).pipe(Effect.provide(makeBase())))

    expect(result.contextLimits?.softCapRatio).toBe(20)
  })
})
