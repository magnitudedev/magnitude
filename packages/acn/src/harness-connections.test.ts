import { Effect, Fiber, Stream, SubscriptionRef } from "effect"
import { describe, expect, it } from "vitest"
import { HarnessIdSchema, type LocalModelsState } from "@magnitudedev/acn-protocol"
import { moveHarnessSelectionsOnAliasChanges } from "./harness-connections"

const alias = "hf:unsloth/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf"
const state = (aliases: readonly string[]) => ({
  models: [{ _tag: "Catalog", modelId: "qwen3.5-4b:gguf:q4", catalogData: { aliases } }],
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
