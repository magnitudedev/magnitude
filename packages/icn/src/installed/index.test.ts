import { Effect, Layer, Ref, Stream, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient, type IcnClientService } from "../client.js"
import { IcnInstalledModels, makeIcnInstalledModels } from "./index.js"
import { IcnEvents } from "../events/index.js"

describe("ICN installed models", () => {
  it("publishes the authoritative installed-package response", async () => {
    const response = {
      revision: 0,
      reconciliationComplete: false,
      packages: [],
    }
    const client = {
      models: { listInstalledModels: () => Effect.succeed(response) },
    } as unknown as IcnClientService
    const result = await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const installed = yield* IcnInstalledModels
          return yield* installed.get
        }).pipe(
          Effect.provide(
            makeIcnInstalledModels().pipe(Layer.provide(
              Layer.merge(
                Layer.succeed(IcnClient, client),
                Layer.succeed(IcnEvents, IcnEvents.of({
                  subscribe: Effect.succeed(Stream.never),
                })),
              ),
            )),
          ),
        ),
      ),
    )

    expect(result).toEqual({
      revision: 1,
      state: response,
    })
  })

  it("refreshes an incomplete startup snapshot until reconciliation completes", async () => {
    const incomplete = {
      revision: 0,
      reconciliationComplete: false,
      packages: [],
    }
    const complete = {
      revision: 1,
      reconciliationComplete: true,
      packages: [],
    }
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const reads = yield* Ref.make(0)
      const client = {
        models: {
          listInstalledModels: () => Ref.getAndUpdate(reads, (count) => count + 1).pipe(
            Effect.map((count) => count === 0 ? incomplete : complete),
          ),
        },
      } as unknown as IcnClientService
      return yield* Effect.gen(function* () {
        const installed = yield* IcnInstalledModels
        yield* Effect.yieldNow()
        yield* TestClock.adjust("1 second")
        yield* Effect.yieldNow()
        return yield* installed.get
      }).pipe(Effect.provide(
        makeIcnInstalledModels({ retryInterval: "1 second" }).pipe(Layer.provide(
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
