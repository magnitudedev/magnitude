import { afterEach, beforeEach, describe, expect, test } from "vitest"
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/ai"
import {
  SlotIdSchema,
  selectedSlotSelection,
  slotSelectionOption,
  unassignedSlotSelection,
} from "../types/config"
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

  test("persists completion atomically without replacing other domains", async () => {
    const base = makeBase()
    const result = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      yield* config.update((current) => ({
        ...current,
        models: {
          slots: {
            primary: selectedSlotSelection(selection("local", "model")),
            secondary: selectedSlotSelection(selection("local", "model")),
          },
          localModelRecency: { primary: [], secondary: [] },
          favoriteModels: [],
          localProviderOfferings: [],
          dismissedDownloadFailures: [],
        },
        futureDomain: { enabled: true },
      }))
      yield* config.updateOnboardingState(true)
      return yield* config.load()
    }).pipe(Effect.provide(base)))

    expect(Option.getOrThrow(Option.flatMap(Option.fromNullable(result.models), (models) =>
      slotSelectionOption(models.slots.primary)))).toEqual(selection("local", "model"))
    const persisted = await Bun.file(makeGlobalStoragePaths(root).configFile).json()
    expect(persisted.futureDomain).toEqual({ enabled: true })
    expect(result.onboarding).toEqual(Option.some({ completed: true }))
    expect(persisted.onboarding).toEqual({ completed: true })
  })

  test("persists incomplete without replacing other configuration", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      yield* config.update((current) => ({
        ...current,
        futureDomain: { enabled: true },
      }))
      yield* config.updateOnboardingState(true)
      yield* config.updateOnboardingState(false)
      return yield* config.load()
    }).pipe(Effect.provide(makeBase())))

    expect(result.onboarding).toEqual(Option.some({ completed: false }))
    const persisted = await Bun.file(makeGlobalStoragePaths(root).configFile).json()
    expect(persisted.futureDomain).toEqual({ enabled: true })
  })

  test("discards invalid onboarding state and incomplete canonical selections", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      models: {
        slots: {
          primary: { providerId: "local", providerModelId: "model" },
        },
      },
      onboarding: {
        unexpected: true,
      },
      futureDomain: { enabled: true },
    }))

    const result = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      const onboarding = yield* config.getOnboardingConfig()
      const loaded = yield* config.load()
      return { onboarding, loaded }
    }).pipe(Effect.provide(makeBase())))

    expect(result.onboarding).toEqual(Option.none())
    expect(Option.map(Option.fromNullable(result.loaded.models), (models) =>
      models.slots.primary)).toEqual(Option.some(unassignedSlotSelection()))
    const persisted = await Bun.file(paths.configFile).json()
    expect(persisted.onboarding).toBeUndefined()
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

    expect(Option.map(Option.fromNullable(models), (value) => value.slots.primary)).toEqual(
      Option.some(unassignedSlotSelection()),
    )
    expect(Option.getOrThrow(Option.flatMap(Option.fromNullable(models), (value) =>
      slotSelectionOption(value.slots.secondary)))).toEqual({
      providerId: "cloud",
      providerModelId: "secondary-model",
      reasoningEffort: "high",
    })
  })

  test("can update onboarding after discarding invalid state", async () => {
    const paths = makeGlobalStoragePaths(root)
    await Bun.write(paths.configFile, JSON.stringify({
      onboarding: { unexpected: true },
      models: { slots: { primary: {
        providerId: "local",
        providerModelId: "local:model",
        reasoningEffort: "high",
      } } },
    }))

    const loaded = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      yield* config.updateOnboardingState(true)
      return yield* config.load()
    }).pipe(Effect.provide(makeBase())))

    expect(loaded.onboarding).toEqual(Option.some({ completed: true }))
    expect(Option.getOrThrow(Option.flatMap(Option.fromNullable(loaded.models), (models) =>
      slotSelectionOption(models.slots.primary))).providerModelId).toBe("local:model")
  })

  test("backs up malformed JSON and resets to the root default", async () => {
    const paths = makeGlobalStoragePaths(root)
    const original = "{ malformed"
    await Bun.write(paths.configFile, original)

    const loaded = await Effect.runPromise(Effect.gen(function* () {
      const config = yield* makeConfigStorage()
      return yield* config.load()
    }).pipe(Effect.provide(makeBase())))

    expect(loaded).toEqual({ onboarding: Option.none(), providers: Option.none() })
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

    expect(loaded).toEqual({ onboarding: Option.none(), providers: Option.none() })
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
    expect(persisted.models.slots.primary).toEqual({
      _tag: "Selected",
      selection: expect.objectContaining({ providerModelId: "model" }),
    })
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
      yield* config.save({
        contextLimits: { softCapRatio: 0, softCapMaxTokens: null },
        onboarding: Option.none(),
        providers: Option.none(),
      })
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
