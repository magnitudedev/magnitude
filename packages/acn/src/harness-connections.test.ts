import { Effect, Fiber, Stream, SubscriptionRef } from "effect"
import { describe, expect, it } from "vitest"
import { HarnessIdSchema, type LocalModelsState } from "@magnitudedev/acn-protocol"
import { isolatedHarnessHome, moveHarnessSelectionsOnAliasChanges } from "./harness-connections"

const alias = "hf:unsloth/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf"
const state = (aliases: readonly string[], installed: readonly string[] = []) => ({
  models: [
    { _tag: "Catalog", modelId: "qwen3.5-4b:gguf:q4", catalogData: { aliases }, acquisitionState: { _tag: "NotInstalled" } },
    ...installed.map((modelId) => ({
      _tag: "Custom",
      modelId,
      acquisitionState: { _tag: "Installed", installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/m.gguf", ownership: "External" }, residencyState: { _tag: "Unloaded" } },
    })),
  ],
  hidden: [],
}) as unknown as LocalModelsState

describe("harness selections after an upgrade", () => {
  it("reconciles connected harnesses at start and whenever the catalog's aliases change, and not otherwise", async () => {
    const runs = await Effect.runPromise(Effect.gen(function* () {
      const models = yield* SubscriptionRef.make(state([alias]))
      const calls: string[] = []
      const fiber = yield* moveHarnessSelectionsOnAliasChanges(
        { state: SubscriptionRef.get(models), changes: models.changes.pipe(Stream.drop(1), Stream.as(undefined)) },
        Effect.sync(() => { calls.push("sync"); return [HarnessIdSchema.make("opencode")] }),
      ).pipe(Effect.fork)
      yield* Effect.sleep("10 millis")
      const atStart = calls.length
      yield* SubscriptionRef.set(models, state([alias]))
      yield* Effect.sleep("10 millis")
      const unchanged = calls.length
      yield* SubscriptionRef.set(models, state([alias, "hf:other/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"]))
      yield* Effect.sleep("10 millis")
      const afterNewAlias = calls.length
      yield* Fiber.interrupt(fiber)
      return { atStart, unchanged, afterNewAlias }
    }))
    expect(runs).toEqual({ atStart: 1, unchanged: 1, afterNewAlias: 2 })
  })

  it("does nothing while the catalog lists no aliases, then reconciles once one appears", async () => {
    const runs = await Effect.runPromise(Effect.gen(function* () {
      const models = yield* SubscriptionRef.make(state([]))
      const calls: string[] = []
      const fiber = yield* moveHarnessSelectionsOnAliasChanges(
        { state: SubscriptionRef.get(models), changes: models.changes.pipe(Stream.drop(1), Stream.as(undefined)) },
        Effect.sync(() => { calls.push("sync"); return [] }),
      ).pipe(Effect.fork)
      yield* Effect.sleep("10 millis")
      const before = calls.length
      yield* SubscriptionRef.set(models, state([alias]))
      yield* Effect.sleep("10 millis")
      const after = calls.length
      yield* Fiber.interrupt(fiber)
      return { before, after }
    }))
    expect(runs).toEqual({ before: 0, after: 1 })
  })
})

describe("harness selections once models are installed", () => {
  it("reconciles again when the engine first has installed models after the aliases were known, and not when nothing changes", async () => {
    const runs = await Effect.runPromise(Effect.gen(function* () {
      const models = yield* SubscriptionRef.make(state([alias]))
      const calls: string[] = []
      const fiber = yield* moveHarnessSelectionsOnAliasChanges(
        { state: SubscriptionRef.get(models), changes: models.changes.pipe(Stream.drop(1), Stream.as(undefined)) },
        Effect.sync(() => { calls.push("sync"); return [] }),
      ).pipe(Effect.fork)
      yield* Effect.sleep("10 millis")
      const beforeModels = calls.length
      yield* SubscriptionRef.set(models, state([alias], ["local:abc"]))
      yield* Effect.sleep("10 millis")
      const afterModels = calls.length
      yield* SubscriptionRef.set(models, state([alias], ["local:abc"]))
      yield* Effect.sleep("10 millis")
      const unchanged = calls.length
      yield* Fiber.interrupt(fiber)
      return { beforeModels, afterModels, unchanged }
    }))
    expect(runs).toEqual({ beforeModels: 1, afterModels: 2, unchanged: 2 })
  })
})

describe("the harness home a service writes to", () => {
  it("is the person's own home for their ~/.magnitude profile on Windows, whatever the path's case or separators", () => {
    expect(isolatedHarnessHome("C:\\Users\\Ada\\.magnitude", "C:\\Users\\ada", "win32")).toBe(false)
    expect(isolatedHarnessHome("c:/users/ada/.magnitude/", "C:\\Users\\Ada", "win32")).toBe(false)
    expect(isolatedHarnessHome("C:\\Users\\Ada\\.magnitude-desktop-dev", "C:\\Users\\Ada", "win32")).toBe(true)
  })

  it("compares exactly elsewhere", () => {
    expect(isolatedHarnessHome("/home/ada/.magnitude", "/home/ada", "linux")).toBe(false)
    expect(isolatedHarnessHome("/home/ada/.magnitude-desktop-dev", "/home/ada", "linux")).toBe(true)
    expect(isolatedHarnessHome("/Users/Ada/.magnitude", "/Users/ada", "darwin")).toBe(true)
  })
})
