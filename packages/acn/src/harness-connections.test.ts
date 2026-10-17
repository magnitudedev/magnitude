import { Effect, Fiber, SubscriptionRef } from "effect"
import { describe, expect, it } from "vitest"
import { HarnessIdSchema, type LocalModelsState } from "@magnitudedev/acn-protocol"
import { isolatedHarnessHome, moveHarnessSelectionsOnAliasChanges } from "./harness-connections"

const alias = "hf:unsloth/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf"
const installation = { _tag: "Resolved", installedBytes: 1, primaryPath: "/m.gguf", ownership: "External" }
/** `offered` models are installed and assessed to fit; `assessing` ones are installed and not assessed yet. */
const state = (aliases: readonly string[], offered: readonly string[] = [], assessing: readonly string[] = []) => ({
  models: [
    { _tag: "Catalog", modelId: "qwen3.5-4b:gguf:q4", catalogData: { aliases, support: { _tag: "Supported" } }, acquisitionState: { _tag: "NotInstalled" }, servingState: { _tag: "Assessing" } },
    ...offered.map((modelId) => ({
      _tag: "Custom",
      modelId,
      acquisitionState: { _tag: "External", installation, residencyState: { _tag: "Unloaded" } },
      servingState: { _tag: "Assessed", assessment: { _tag: "Fits" } },
    })),
    ...assessing.map((modelId) => ({
      _tag: "Custom",
      modelId,
      acquisitionState: { _tag: "External", installation, residencyState: { _tag: "Unloaded" } },
      servingState: { _tag: "Assessing" },
    })),
  ],
  hidden: [],
}) as unknown as LocalModelsState

/** Runs the move over a state that `steps` change in turn, and counts the syncs after the start and after each step. */
const syncsAfter = (initial: LocalModelsState, steps: readonly LocalModelsState[]) => Effect.runPromise(Effect.gen(function* () {
  const models = yield* SubscriptionRef.make(initial)
  let calls = 0
  const fiber = yield* moveHarnessSelectionsOnAliasChanges(
    models.changes,
    Effect.sync(() => { calls += 1; return [HarnessIdSchema.make("opencode")] }),
  ).pipe(Effect.fork)
  yield* Effect.sleep("10 millis")
  const counts = [calls]
  for (const step of steps) {
    yield* SubscriptionRef.set(models, step)
    yield* Effect.sleep("10 millis")
    counts.push(calls)
  }
  yield* Fiber.interrupt(fiber)
  return counts
}))

describe("harness selections after an upgrade", () => {
  it("reconciles at start when aliases and offered models are known, and again only when the aliases change", async () => {
    expect(await syncsAfter(state([alias], ["local:abc"]), [
      state([alias], ["local:abc"]),
      state([alias, "hf:other/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"], ["local:abc"]),
    ])).toEqual([1, 1, 2])
  })

  it("waits for the aliases when the models are offered first", async () => {
    expect(await syncsAfter(state([], ["local:abc"]), [state([alias], ["local:abc"])])).toEqual([0, 1])
  })

  it("waits for the new engine to assess an installed model before reconciling, so the service offers it", async () => {
    expect(await syncsAfter(state([alias], [], ["local:abc"]), [
      state([alias], [], ["local:abc"]),
      state([alias], ["local:abc"]),
      state([alias], ["local:abc"]),
    ])).toEqual([0, 0, 1, 1])
  })

  it("waits for models to arrive after the service starts with none", async () => {
    expect(await syncsAfter(state([alias]), [state([alias], [], ["local:abc"]), state([alias], ["local:abc"])])).toEqual([0, 0, 1])
  })

  it("reconciles again when another model becomes offered, so a selection whose model wasn't offered yet moves too", async () => {
    expect(await syncsAfter(state([alias], ["local:abc"]), [state([alias], ["local:abc", "qwen3.5-4b:gguf:q4"])])).toEqual([1, 2])
  })

  it("misses no change made while it starts", async () => {
    const calls = await Effect.runPromise(Effect.gen(function* () {
      const models = yield* SubscriptionRef.make(state([alias]))
      let calls = 0
      const fiber = yield* moveHarnessSelectionsOnAliasChanges(
        models.changes,
        Effect.sync(() => { calls += 1; return [] }),
      ).pipe(Effect.fork)
      yield* SubscriptionRef.set(models, state([alias], ["local:abc"]))
      yield* Effect.sleep("10 millis")
      yield* Fiber.interrupt(fiber)
      return calls
    }))
    expect(calls).toBe(1)
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
