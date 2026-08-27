import { Effect, Layer, Ref, Stream, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient, type IcnClientService } from "../client.js"
import { IcnEvents } from "../events/index.js"
import { IcnModels, makeIcnModels } from "./index.js"

describe("ICN models", () => {
  it("refreshes an incomplete startup snapshot until reconciliation completes", async () => {
    const incomplete = {
      revision: 0,
      reconciliationComplete: false,
      models: [],
      diagnostics: [],
    }
    const complete = {
      ...incomplete,
      revision: 1,
      reconciliationComplete: true,
    }
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const reads = yield* Ref.make(0)
      const client = {
        models: {
          listModels: () => Ref.getAndUpdate(reads, (count) => count + 1).pipe(
            Effect.flatMap((count) => count === 0
              ? Effect.succeed(incomplete)
              : count === 1
                ? Effect.fail("transient startup read failure")
                : Effect.succeed(complete)),
          ),
        },
      } as unknown as IcnClientService
      return yield* Effect.gen(function* () {
        const models = yield* IcnModels
        yield* Effect.yieldNow()
        yield* TestClock.adjust("3 seconds")
        yield* Effect.yieldNow()
        return yield* models.get
      }).pipe(Effect.provide(
        makeIcnModels({ retryInterval: "1 second" }).pipe(Layer.provide(
          Layer.merge(
            Layer.succeed(IcnClient, client),
            Layer.succeed(IcnEvents, IcnEvents.of({
              subscribe: Effect.succeed(Stream.never),
            })),
          ),
        )),
      ))
    })).pipe(Effect.provide(TestContext.TestContext)))

    expect(result).toEqual({
      revision: 2,
      state: complete,
    })
  })
})
