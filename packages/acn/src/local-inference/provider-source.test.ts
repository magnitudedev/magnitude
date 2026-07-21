import { describe, expect, it } from "vitest"
import { FetchHttpClient } from "@effect/platform"
import * as Headers from "@effect/platform/Headers"
import { Effect, Either, Fiber, Layer, Stream } from "effect"
import {
  IcnApiClient,
  GeneratedClientRemoteError,
  type IcnApiClient as IcnApiClientService,
} from "@magnitudedev/icn"
import { AcnActivityTracker, AcnActivityTrackerLive } from "../activity-tracker"
import { LocalModelInventoryChanges } from "./inventory-changes"
import { LocalModelConfiguration, type LocalModelConfigurationApi } from "./model-configuration"
import {
  LocalModelProviderSource,
  LocalModelProviderSourceLive,
} from "./provider-source"

const emptyConfiguration = Layer.succeed(LocalModelConfiguration, {
  get: Effect.succeed({}),
  getModels: Effect.succeed({}),
  selectProfile: () => Effect.void,
  updateSlots: () => Effect.void,
  recordUse: () => Effect.void,
  revision: Effect.succeed(0),
  changes: Stream.empty,
} satisfies LocalModelConfigurationApi)

describe("local model provider catalog", () => {
  it("caches list, forces refresh, and invalidates on inventory changes", async () => {
    let listCalls = 0
    let revision = 0
    const client = {
      models: {
        listModels: () => Effect.sync(() => {
          listCalls++
          return { object: "list", data: [] }
        }),
      },
      system: {
        health: () => Effect.succeed({ ready: true, status: "ok" }),
      },
    } as unknown as IcnApiClientService
    const test = Effect.gen(function* () {
      const source = yield* LocalModelProviderSource
      yield* source.catalog.list
      yield* source.catalog.list
      expect(listCalls).toBe(1)

      yield* source.catalog.refresh
      expect(listCalls).toBe(2)

      revision++
      yield* source.catalog.list
      expect(listCalls).toBe(3)
    }).pipe(Effect.provide(LocalModelProviderSourceLive.pipe(Layer.provide(Layer.mergeAll(
      Layer.succeed(IcnApiClient, client),
      AcnActivityTrackerLive,
      emptyConfiguration,
      Layer.succeed(LocalModelInventoryChanges, {
        publish: Effect.sync(() => { revision++ }),
        revision: Effect.sync(() => revision),
        stream: Stream.empty,
      }),
    )))))

    await Effect.runPromise(test.pipe(Effect.provide(FetchHttpClient.layer)))
  })

  it("preserves an ICN HTTP rejection as a stream-start provider rejection", async () => {
    const client = {
      chat: {
        createChatCompletion: () => Effect.fail(new GeneratedClientRemoteError({
          operationId: "createChatCompletion",
          status: 400,
          headers: Headers.fromInput({ "content-type": "application/json" }),
          body: {
            error: {
              message: "assistant messages require content, reasoning_content, or tool_calls",
              type: "invalid_request_error",
              code: "invalid_request",
            },
          },
        })),
      },
      models: { listModels: () => Effect.succeed({ object: "list", data: [] }) },
      system: { health: () => Effect.succeed({ ready: true, status: "ok" }) },
    } as unknown as IcnApiClientService
    const dependencies = Layer.mergeAll(
      Layer.succeed(IcnApiClient, client),
      AcnActivityTrackerLive,
      emptyConfiguration,
      Layer.succeed(LocalModelInventoryChanges, {
        publish: Effect.void,
        revision: Effect.succeed(0),
        stream: Stream.empty,
      }),
    )
    const layer = Layer.merge(dependencies, LocalModelProviderSourceLive.pipe(Layer.provide(dependencies)))
    const result = await Effect.runPromise(Effect.gen(function* () {
      const source = yield* LocalModelProviderSource
      const bound = yield* source.bindModel("model-1" as never)
      return yield* Effect.either(bound.stream({
        system: "You are helpful.",
        messages: [{ _tag: "UserMessage", parts: [{ _tag: "TextPart", text: "hello" }] }],
      } as never, []))
    }).pipe(Effect.provide(Layer.merge(layer, FetchHttpClient.layer))))

    expect(Either.isLeft(result)).toBe(true)
    if (Either.isLeft(result)) {
      expect(result.left._tag).toBe("StreamStartProviderRejection")
      if (result.left._tag === "StreamStartProviderRejection") {
        expect(result.left.response.status).toBe(400)
        expect(result.left.rejection.message).toContain("assistant messages require content")
      }
    }
  })

  it("releases active work when chat admission is interrupted", async () => {
    const client = {
      chat: { createChatCompletion: () => Effect.never },
      models: { listModels: () => Effect.succeed({ object: "list", data: [] }) },
      system: { health: () => Effect.succeed({ ready: true, status: "ok" }) },
    } as unknown as IcnApiClientService
    const dependencies = Layer.mergeAll(
      Layer.succeed(IcnApiClient, client),
      AcnActivityTrackerLive,
      emptyConfiguration,
      Layer.succeed(LocalModelInventoryChanges, {
        publish: Effect.void,
        revision: Effect.succeed(0),
        stream: Stream.empty,
      }),
    )
    const layer = Layer.merge(dependencies, LocalModelProviderSourceLive.pipe(Layer.provide(dependencies)))

    await Effect.runPromise(Effect.gen(function* () {
      const source = yield* LocalModelProviderSource
      const activity = yield* AcnActivityTracker
      const bound = yield* source.bindModel("model-1" as never)
      const admission = yield* Effect.fork(bound.stream({
        system: "You are helpful.",
        messages: [{ _tag: "UserMessage", parts: [{ _tag: "TextPart", text: "hello" }] }],
      } as never, []))
      yield* Effect.yieldNow()
      expect(yield* activity.hasActiveWork).toBe(true)
      yield* Fiber.interrupt(admission)
      expect(yield* activity.hasActiveWork).toBe(false)
    }).pipe(Effect.provide(Layer.merge(layer, FetchHttpClient.layer))))
  })
})
