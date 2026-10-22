import { Effect, Exit, Layer, Ref, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient, type IcnClientService } from "../client.js"
import { IcnEvents } from "../events/index.js"
import {
  IcnCatalog,
  IcnLocalModels,
  IcnModelAssessments,
  makeIcnCatalog,
  makeIcnLocalModels,
  makeIcnModelAssessments,
} from "./index.js"

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

describe("ICN local models", () => {
  const empty = { revision: 0, reconciliationComplete: false, models: [], unsupported: [] }

  it("exposes local models immediately without requesting reconciliation", async () => {
    const reconciliations = { count: 0 }
    const client = { localModels: {
      listLocalModels: () => Effect.succeed(empty),
      refreshLocalModels: () => Effect.sync(() => {
        reconciliations.count += 1
        return empty
      }),
    } } as unknown as IcnClientService
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const localModels = yield* IcnLocalModels
      return yield* localModels.get
    }).pipe(Effect.provide(makeIcnLocalModels().pipe(Layer.provide(
      Layer.merge(Layer.succeed(IcnClient, client), eventsLayer),
    ))))))
    expect(result).toEqual({ revision: 1, state: empty })
    expect(reconciliations.count).toBe(0)
  })

  it("fails an explicit reconciliation instead of waiting forever", async () => {
    const client = { localModels: {
      listLocalModels: () => Effect.succeed({ ...empty, reconciliationComplete: true }),
      refreshLocalModels: () => Effect.never,
    } } as unknown as IcnClientService
    const program = Effect.gen(function* () {
      const localModels = yield* IcnLocalModels
      return yield* localModels.reconcile.pipe(Effect.exit)
    }).pipe(Effect.provide(makeIcnLocalModels({ reconciliationTimeout: "10 millis" }).pipe(Layer.provide(
      Layer.merge(Layer.succeed(IcnClient, client), eventsLayer),
    ))))
    const exit = await Effect.runPromise(Effect.scoped(program))
    expect(Exit.isFailure(exit)).toBe(true)
  })

  it("replaces the whole configuration and rereads the published models", async () => {
    const configured: unknown[] = []
    const reconciled = { ...empty, revision: 3, reconciliationComplete: true }
    const client = { localModels: {
      listLocalModels: () => Effect.sync(() => configured.length === 0 ? empty : reconciled),
      refreshLocalModels: () => Effect.succeed(reconciled),
      configureLocalModels: ({ payload }: { readonly payload: unknown }) => Effect.sync(() => {
        configured.push(payload)
        return payload
      }),
    } } as unknown as IcnClientService
    const configuration = {
      roots: [{ kind: "Folder" as const, path: "/models", enabled: true }],
      added: [],
      hidden: [],
      drafters: [],
    }
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const localModels = yield* IcnLocalModels
      yield* localModels.configure(configuration)
      return (yield* localModels.get).state
    }).pipe(Effect.provide(makeIcnLocalModels().pipe(Layer.provide(
      Layer.merge(Layer.succeed(IcnClient, client), eventsLayer),
    ))))))
    expect(configured).toEqual([configuration])
    expect(result).toEqual(reconciled)
  })
})

describe("ICN automatic model assessments", () => {
  it("observes the read-only assessment snapshot without starting work", async () => {
    const snapshot = {
      revision: 4,
      environmentId: "environment",
      catalog: { _tag: "Pending" as const, sourceRevision: 0 },
      custom: { _tag: "Pending" as const, sourceRevision: 0 },
    }
    const reads = { count: 0 }
    const client = { models: {
      getModelAssessments: () => Effect.sync(() => {
        reads.count += 1
        return snapshot
      }),
    } } as unknown as IcnClientService
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const assessments = yield* IcnModelAssessments
      return yield* assessments.get
    }).pipe(Effect.provide(makeIcnModelAssessments().pipe(Layer.provide(
      Layer.merge(Layer.succeed(IcnClient, client), eventsLayer),
    ))))))
    expect(result).toEqual({ revision: 1, state: snapshot })
    expect(reads.count).toBe(1)
  })
})
