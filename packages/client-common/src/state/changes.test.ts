import { Registry } from "@effect-atom/atom-react"
import { Client, Subscription } from "@magnitudedev/effect-query"
import { Duration, Effect, Fiber, Layer, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { reopenDelay, reopenEndedSubscription } from "./changes"

const waitFor = (condition: () => boolean) => Effect.gen(function* () {
  for (let tries = 0; tries < 200 && !condition(); tries += 1) yield* Effect.sleep("10 millis")
  expect(condition()).toBe(true)
})

describe("the change subscription", () => {
  it("reopens after the stream dies or ends, and has the cache reread each time", async () => {
    const registry = Registry.make()
    const client = Client.make(Layer.empty)
    let opened = 0
    const changes = Subscription.make("StreamChanges", {
      key: (_: void) => "changes",
      stream: () => Stream.suspend(() => {
        opened += 1
        if (opened === 1) return Stream.die(new Error("the service ended the connection"))
        if (opened === 2) return Stream.make("poke")
        return Stream.make("poke").pipe(Stream.concat(Stream.never))
      }),
    })
    const atom = client.subscription(changes, undefined)
    let rereads = 0
    await Effect.runPromise(Effect.gen(function* () {
      const reopening = yield* reopenEndedSubscription(registry, atom, Effect.sync(() => { rereads += 1 })).pipe(Effect.fork)
      yield* waitFor(() => opened === 3 && registry.get(atom).status === "active")
      expect(rereads).toBe(2)
      yield* Effect.sleep("600 millis")
      expect(opened).toBe(3)
      yield* Fiber.interrupt(reopening)
    }))
    registry.dispose()
  })

  it("backs off briefly, doubling up to five seconds", () => {
    expect([0, 1, 2, 10].map((endings) => Duration.toMillis(reopenDelay(endings)))).toEqual([250, 500, 1_000, 5_000])
  })
})
