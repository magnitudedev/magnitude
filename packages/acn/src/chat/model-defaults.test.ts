import { BunContext } from "@effect/platform-bun"
import { Chunk, Effect, Layer, Option, Stream } from "effect"
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { ServingModelId, type ChatModelDefaults } from "@magnitudedev/acn-protocol"
import { LOOPBACK_ONLY } from "@magnitudedev/storage"
import { AcnChanges, AcnChangesLive } from "../changes"
import { AcnHost } from "../server-settings"
import { ChatModelDefaultsStore, ChatModelDefaultsStoreLive } from "./model-defaults"

const layerFor = (dataDir: string) => ChatModelDefaultsStoreLive.pipe(
  Layer.provideMerge(AcnChangesLive),
  Layer.provide(Layer.succeed(AcnHost, { dataDir, port: 10100, activeNetwork: LOOPBACK_ONLY, container: Option.none() })),
  Layer.provide(BunContext.layer),
)
const run = <A, E>(dataDir: string, effect: Effect.Effect<A, E, ChatModelDefaultsStore | AcnChanges>) =>
  Effect.runPromise(Effect.scoped(effect.pipe(Effect.provide(layerFor(dataDir)))))
const fresh = () => mkdtempSync(join(tmpdir(), "magnitude-chat-model-defaults-"))

const qwen = ServingModelId.make("qwen3.5-4b:gguf:q4")
const lfm = ServingModelId.make("lfm3-1.7b:gguf:q8")
const thinkingOff: ChatModelDefaults = {
  reasoningEffort: Option.some("none"),
  settings: {
    systemPrompt: "Answer briefly.",
    temperature: Option.some(0.7),
    maxOutputTokens: Option.none(),
    topP: Option.some(0.8),
    topK: Option.some(20),
    minP: Option.some(0),
    repetitionPenalty: Option.some(1),
    presencePenalty: Option.some(1.5),
    frequencyPenalty: Option.none(),
    seed: Option.none(),
    stop: ["</answer>"],
  },
}

describe("chat model defaults", () => {
  it("has none until one is saved", async () => {
    expect(await run(fresh(), Effect.flatMap(ChatModelDefaultsStore, (store) => store.list))).toEqual({})
  })

  it("keeps each model's defaults, including thinking off, for the next start", async () => {
    const dataDir = fresh()
    await run(dataDir, Effect.flatMap(ChatModelDefaultsStore, (store) => store.save(qwen, thinkingOff)))
    const saved = await run(dataDir, Effect.flatMap(ChatModelDefaultsStore, (store) => store.list))
    expect(saved).toEqual({ [qwen]: thinkingOff })
  })

  it("saving one model leaves the others", async () => {
    const saved = await run(fresh(), Effect.gen(function* () {
      const store = yield* ChatModelDefaultsStore
      yield* store.save(qwen, thinkingOff)
      yield* store.save(lfm, { ...thinkingOff, reasoningEffort: Option.some("high") })
      return yield* store.list
    }))
    expect(Object.keys(saved).sort()).toEqual([lfm, qwen].sort())
  })

  it("clearing removes only that model", async () => {
    const saved = await run(fresh(), Effect.gen(function* () {
      const store = yield* ChatModelDefaultsStore
      yield* store.save(qwen, thinkingOff)
      yield* store.save(lfm, thinkingOff)
      yield* store.clear(qwen)
      return yield* store.list
    }))
    expect(Object.keys(saved)).toEqual([lfm])
  })

  it("keeps the rest of config.json as it was", async () => {
    const dataDir = fresh()
    writeFileSync(join(dataDir, "config.json"), JSON.stringify({ appearance: "dark" }))
    await run(dataDir, Effect.flatMap(ChatModelDefaultsStore, (store) => store.save(qwen, thinkingOff)))
    expect(JSON.parse(readFileSync(join(dataDir, "config.json"), "utf8")).appearance).toBe("dark")
  })

  it("skips an entry it can't read and keeps the others", async () => {
    const dataDir = fresh()
    writeFileSync(join(dataDir, "config.json"), JSON.stringify({ chatModelDefaults: { [lfm]: { settings: { temperature: "hot" } } } }))
    const saved = await run(dataDir, Effect.gen(function* () {
      const store = yield* ChatModelDefaultsStore
      yield* store.save(qwen, thinkingOff)
      return yield* store.list
    }))
    expect(saved).toEqual({ [qwen]: thinkingOff })
    expect(JSON.parse(readFileSync(join(dataDir, "config.json"), "utf8")).chatModelDefaults[lfm]).toBeDefined()
  })

  it("tells other clients the defaults changed", async () => {
    const pokes = await run(fresh(), Effect.gen(function* () {
      const changes = yield* AcnChanges
      const store = yield* ChatModelDefaultsStore
      const received = yield* changes.stream.pipe(Stream.take(1), Stream.runCollect, Effect.fork)
      yield* Effect.sleep("10 millis")
      yield* store.save(qwen, thinkingOff)
      return Chunk.toReadonlyArray(yield* received.await.pipe(Effect.flatten))
    }))
    expect(pokes).toEqual([{ operation: "ListChatModelDefaults" }])
  })
})
