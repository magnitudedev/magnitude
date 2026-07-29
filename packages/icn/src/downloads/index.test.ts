import { Effect, Layer, Ref } from "effect"
import { describe, expect, it } from "vitest"
import { IcnClient, type IcnClientService } from "../client.js"
import type { DownloadAttempt } from "@magnitudedev/icn-protocol/schemas"
import { IcnDownloads, makeIcnDownloads } from "./index.js"

const pending: DownloadAttempt = {
  _tag: "Pending",
  id: "download-test",
  packageId: "package-test",
}

describe("ICN downloads", () => {
  it("publishes an admitted attempt immediately without listing downloads again", async () => {
    const reads = await Effect.runPromise(Ref.make(0))
    const client = {
      models: {
        listModelDownloads: () => Ref.updateAndGet(reads, (count) => count + 1).pipe(
          Effect.as({ attempts: [] }),
        ),
      },
    } as unknown as IcnClientService

    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const downloads = yield* IcnDownloads
      yield* downloads.observeAttempt(pending)
      yield* downloads.observeAttempt({
        ...pending,
        _tag: "Completed",
      })
      return {
        reads: yield* Ref.get(reads),
        snapshot: yield* downloads.get,
      }
    }).pipe(
      Effect.provide(
        makeIcnDownloads({
          refreshInterval: "1 hour",
          idleRefreshInterval: "1 hour",
        }).pipe(
          Layer.provide(Layer.succeed(IcnClient, client)),
        ),
      ),
    )))

    expect(result).toEqual({
      reads: 1,
      snapshot: {
        revision: 2,
        state: {
          attempts: [{
            ...pending,
            _tag: "Completed",
          }],
        },
      },
    })
  })
})
