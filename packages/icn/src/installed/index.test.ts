import { Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { IcnModels } from "../models/index.js"
import { IcnInstalledModels, makeIcnInstalledModels } from "./index.js"

describe("ICN installed models", () => {
  it("does not block service startup on the initial inventory refresh", async () => {
    const result = await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const installed = yield* IcnInstalledModels
          return yield* installed.get
        }).pipe(
          Effect.provide(
            makeIcnInstalledModels().pipe(Layer.provide(
              Layer.succeed(IcnModels, IcnModels.of({
                get: Effect.succeed({
                  revision: 0,
                  state: {
                    revision: 0,
                    reconciliationComplete: false,
                    catalogModels: [],
                    uncataloguedPackages: [],
                    diagnostics: [],
                  },
                }),
                changes: Stream.never,
                initialized: Effect.succeed(false),
                refresh: Effect.never,
                reconcileCatalogModel: () => Effect.never,
              })),
            )),
          ),
          Effect.timeoutOption("1 second"),
        ),
      ),
    )

    expect(Option.getOrThrow(result)).toEqual({
      revision: 0,
      state: {
        revision: 0,
        reconciliationComplete: false,
        packages: [],
      },
    })
  })
})
