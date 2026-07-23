import { Effect, Layer, Option } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient, type IcnClientService } from "../client.js"
import { IcnInstalledModels, makeIcnInstalledModels } from "./index.js"

describe("ICN installed models", () => {
  it("does not block service startup on the initial inventory refresh", async () => {
    const client = {
      models: {
        listInstalledModels: () => Effect.never,
      },
    } as unknown as IcnClientService

    const result = await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const installed = yield* IcnInstalledModels
          return yield* installed.get
        }).pipe(
          Effect.provide(
            makeIcnInstalledModels().pipe(
              Layer.provide(Layer.succeed(IcnClient, client)),
            ),
          ),
          Effect.timeoutOption("1 second"),
        ),
      ),
    )

    expect(Option.getOrThrow(result)).toEqual({
      revision: 0,
      state: { packages: [] },
    })
  })
})
