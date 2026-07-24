import { afterEach, beforeEach, describe, expect, test } from "vitest"
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/ai"
import { SlotIdSchema } from "../types/config"
import { makeGlobalStoragePaths } from "../paths"
import { GlobalStorage } from "../services"
import { makeConfigStorage } from "./storage"

const selection = (providerId: string, providerModelId: string) => ({
  providerId: ProviderIdSchema.make(providerId),
  providerModelId: ProviderModelIdSchema.make(providerModelId),
  reasoningEffort: ReasoningEffortSchema.make("high"),
})

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
            primary: Option.some(selection("local", "model")),
            secondary: Option.some(selection("local", "model")),
          },
          localModelRecency: { primary: [], secondary: [] },
          favoriteModels: [],
          localProviderOfferings: [],
          dismissedDownloadFailures: [],
        },
        futureDomain: { enabled: true },
      }))
      yield* config.completeOnboardingFlow("model_setup", 2, "2026-07-14T22:00:00.000Z")
      return yield* config.load()
    }).pipe(Effect.provide(base)))

    expect(Option.getOrThrow(Option.flatMap(Option.fromNullable(result.models), (models) =>
      models.slots.primary))).toEqual(selection("local", "model"))
    const persisted = await Bun.file(makeGlobalStoragePaths(root).configFile).json()
    expect(persisted.futureDomain).toEqual({ enabled: true })
    expect(result.onboarding?.completions?.model_setup).toEqual({
      version: 2,
      completedAt: "2026-07-14T22:00:00.000Z",
    })
  })

  test("recovers stale onboarding and discards incomplete canonical selections", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      models: {
        slots: {
          primary: { providerId: "local", providerModelId: "model" },
        },
      },
      onboarding: {
        completions: {
          model_setup: { completedAt: "2026-07-14T22:00:00.000Z" },
        },
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
    expect(Option.flatMap(Option.fromNullable(result.loaded.models), (models) =>
      models.slots.primary)).toEqual(Option.none())
    const persisted = await Bun.file(paths.configFile).json()
    expect(persisted.onboarding?.completions?.model_setup).toBeUndefined()
    expect(persisted.futureDomain).toEqual({ enabled: true })
  })

  test("discards an invalid model-slot leaf without replacing a valid sibling slot", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      models: {
        slots: {
          primary: { providerId: 42, providerModelId: "primary-model" },
          secondary: { providerId: "cloud", providerModelId: "secondary-model", reasoningEffort: "high" },
        },
      },
    }))

    const models = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      return (yield* config.load()).models ?? null
    }).pipe(Effect.provide(makeBase())))

    expect(Option.flatMap(Option.fromNullable(models), (value) => value.slots.primary)).toEqual(Option.none())
    expect(Option.getOrThrow(Option.flatMap(Option.fromNullable(models), (value) =>
      value.slots.secondary))).toEqual({
      providerId: "cloud",
      providerModelId: "secondary-model",
      reasoningEffort: "high",
    })
  })

  test("can complete onboarding after recovering stale state", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      onboarding: { completions: { model_setup: { completedAt: "old" } } },
      models: { slots: { primary: {
        providerId: "local",
        providerModelId: "local:model",
        reasoningEffort: "high",
      } } },
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
    expect(Option.getOrThrow(Option.flatMap(Option.fromNullable(loaded.models), (models) =>
      models.slots.primary)).providerModelId).toBe("local:model")
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
      yield* config.updateModelSlot(
        SlotIdSchema.make("primary"),
        Option.some(selection("local", "model")),
      )
    }).pipe(Effect.provide(makeBase())))

    const persisted = await Bun.file(paths.configFile).json()
    expect(persisted.futureDomain).toEqual({ value: 42 })
    expect(persisted.models.slots.primary.providerModelId).toBe("model")
  })

  test("discards removed model configuration without erasing unrelated unknown fields", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      futureDomain: { value: 42 },
      localInference: { selectedProfile: { configurationId: "legacy" } },
      models: {
        slots: {},
        localSlotIntent: { primary: "local" },
        localModelRecency: { primary: ["local:recent"], secondary: [] },
      },
    }))

    await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      yield* config.load()
    }).pipe(Effect.provide(makeBase())))

    const persisted = await Bun.file(paths.configFile).json()
    expect(persisted).not.toHaveProperty("localInference")
    expect(persisted.models).not.toHaveProperty("localSlotIntent")
    expect(persisted.models.localModelRecency).toEqual({ primary: ["local:recent"], secondary: [] })
    expect(persisted.futureDomain).toEqual({ value: 42 })
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
