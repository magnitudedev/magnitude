import { Effect, Exit, Layer, Ref, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient, type IcnClientService } from "../client.js"
import { IcnEvents } from "../events/index.js"
import { IcnCatalog, IcnDiscovery, makeIcnCatalog, makeIcnDiscovery } from "./index.js"

const eventsLayer = Layer.succeed(
  IcnEvents,
  IcnEvents.of({ subscribe: Effect.succeed(Stream.never) }),
)

describe("ICN catalog", () => {
  it("exposes an incomplete snapshot without opening an immortal polling loop", async () => {
    const incomplete = { revision: 0, reconciliationComplete: false, models: [] }
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const reads = yield* Ref.make(0)
      const client = { catalog: { listCatalogModels: () => Ref.update(reads, (count) => count + 1).pipe(
        Effect.as(incomplete),
      ) } } as unknown as IcnClientService
      return yield* Effect.gen(function* () {
        const catalog = yield* IcnCatalog
        return { state: yield* catalog.get, reads: yield* Ref.get(reads) }
      }).pipe(Effect.provide(makeIcnCatalog({ retryInterval: "1 second" }).pipe(Layer.provide(
        Layer.merge(
          Layer.succeed(IcnClient, client),
          eventsLayer,
        ),
      ))))
    })))
    expect(result).toEqual({ state: { revision: 1, state: incomplete }, reads: 1 })
  })
})

describe("ICN discovery", () => {
  it("completes initial reconciliation before exposing discovery state", async () => {
    const incomplete = { revision: 0, reconciliationComplete: false, models: [] }
    const complete = { revision: 1, reconciliationComplete: true, models: [] }
    const reconciliations = { count: 0 }
    const client = { discovery: {
      listDiscoveredModels: () => Effect.succeed(incomplete),
      refreshDiscoveredModels: () => Effect.sync(() => {
        reconciliations.count += 1
        return complete
      }),
    } } as unknown as IcnClientService
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const discovery = yield* IcnDiscovery
      return yield* discovery.get
    }).pipe(Effect.provide(makeIcnDiscovery().pipe(Layer.provide(
      Layer.merge(Layer.succeed(IcnClient, client), eventsLayer),
    ))))))
    expect(result).toEqual({ revision: 1, state: complete })
    expect(reconciliations.count).toBe(1)
  })

  it("fails startup when initial reconciliation fails", async () => {
    const incomplete = { revision: 0, reconciliationComplete: false, models: [] }
    const client = { discovery: {
      listDiscoveredModels: () => Effect.succeed(incomplete),
      refreshDiscoveredModels: () => Effect.fail("inventory scan failed"),
    } } as unknown as IcnClientService
    const exit = await Effect.runPromise(Effect.scoped(IcnDiscovery.pipe(
      Effect.provide(makeIcnDiscovery().pipe(Layer.provide(
        Layer.merge(Layer.succeed(IcnClient, client), eventsLayer),
      ))),
      Effect.exit,
    )))
    expect(Exit.isFailure(exit)).toBe(true)
  })

  it("fails startup instead of waiting forever for initial reconciliation", async () => {
    const incomplete = { revision: 0, reconciliationComplete: false, models: [] }
    const client = { discovery: {
      listDiscoveredModels: () => Effect.succeed(incomplete),
      refreshDiscoveredModels: () => Effect.never,
    } } as unknown as IcnClientService
    const exit = await Effect.runPromise(Effect.scoped(IcnDiscovery.pipe(
      Effect.provide(makeIcnDiscovery({ reconciliationTimeout: "10 millis" }).pipe(Layer.provide(
        Layer.merge(Layer.succeed(IcnClient, client), eventsLayer),
      ))),
      Effect.exit,
    )))
    expect(Exit.isFailure(exit)).toBe(true)
  })

  it("fails an explicit reconciliation instead of waiting forever", async () => {
    const complete = { revision: 1, reconciliationComplete: true, models: [] }
    const client = { discovery: {
      listDiscoveredModels: () => Effect.succeed(complete),
      refreshDiscoveredModels: () => Effect.never,
    } } as unknown as IcnClientService
    const program = Effect.gen(function* () {
      const discovery = yield* IcnDiscovery
      return yield* discovery.reconcile.pipe(Effect.exit)
    }).pipe(Effect.provide(makeIcnDiscovery({ reconciliationTimeout: "10 millis" }).pipe(Layer.provide(
      Layer.merge(Layer.succeed(IcnClient, client), eventsLayer),
    ))))
    const exit = await Effect.runPromise(Effect.scoped(program))
    expect(Exit.isFailure(exit)).toBe(true)
  })
})
