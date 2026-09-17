import { Deferred, Effect, Fiber } from "effect"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { MacUpdateStageFailed, NativeMacUpdate, stageMacUpdateArchive } from "./mac-update-stage"

describe("Mac update staging endpoint", () => {
  it("serves only the exact archive through a private feed and closes after staging", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-stage-test-"))
    const file = join(root, "verified.zip")
    await writeFile(file, "verified archive bytes")
    let endpoint = ""
    try {
      await Effect.runPromise(stageMacUpdateArchive(file).pipe(Effect.provideService(NativeMacUpdate, {
        stage: feed => Effect.promise(async () => {
          endpoint = feed.url
          expect(new URL(endpoint).hostname).toBe("127.0.0.1")
          expect((await fetch(endpoint)).status).toBe(404)
          const response = await fetch(endpoint, { headers: { Authorization: feed.authorization } })
          expect(response.status).toBe(200)
          const { url } = await response.json() as { url: string }
          expect(new URL(url).origin).toBe(new URL(endpoint).origin)
          const archive = await fetch(url)
          expect(archive.headers.get("content-type")).toBe("application/zip")
          expect(await archive.text()).toBe("verified archive bytes")
          expect((await fetch(new URL("/verified.zip", endpoint))).status).toBe(404)
          expect((await fetch(url, { method: "POST" })).status).toBe(405)
        }),
      })))
      await expect(fetch(endpoint)).rejects.toThrow()
    } finally { await rm(root, { recursive: true, force: true }) }
  })

  it("preserves native staging failure and closes its endpoint", async () => {
    let endpoint = ""
    const result = await Effect.runPromise(stageMacUpdateArchive("/unused.zip").pipe(
      Effect.provideService(NativeMacUpdate, {
        stage: feed => Effect.suspend(() => {
          endpoint = feed.url
          return new MacUpdateStageFailed({ message: "Native signature verification failed" })
        }),
      }), Effect.flip,
    ))
    expect(result.message).toBe("Native signature verification failed")
    await expect(fetch(endpoint)).rejects.toThrow()
  })

  it("closes the endpoint and cancels native observation when its owner exits", async () => {
    const endpoint = await Effect.runPromise(Effect.gen(function* () {
      const started = yield* Deferred.make<string>()
      const observationClosed = yield* Deferred.make<void>()
      const staging = yield* stageMacUpdateArchive("/unused.zip").pipe(
        Effect.provideService(NativeMacUpdate, {
          stage: feed => Deferred.succeed(started, feed.url).pipe(
            Effect.zipRight(Effect.never),
            Effect.ensuring(Deferred.succeed(observationClosed, undefined)),
          ),
        }), Effect.forkScoped,
      )
      const url = yield* Deferred.await(started)
      yield* Effect.promise(async () => expect((await fetch(url)).status).toBe(404))
      yield* Fiber.interrupt(staging)
      yield* Deferred.await(observationClosed)
      return url
    }).pipe(Effect.scoped))
    await expect(fetch(endpoint)).rejects.toThrow()
  })
})
