import { Effect, Stream, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import type { ApplicationUpdate } from "./application-update"
import { makeHeadlessUpdateControl } from "./headless-update"

const fixture = () => {
  const calls: string[] = []
  const call = (name: string) => Effect.sync(() => { calls.push(name) })
  const state = { transfer: { _tag: "Ready", version: "0.1.6" }, check: { _tag: "Idle" }, preference: { _tag: "Known", autoDownload: true } } as const
  const updates: ApplicationUpdate = { state: Effect.succeed(state), changes: Stream.succeed(state),
    check: call("check"), download: call("download"), discard: call("discard"), close: call("close"),
    requireReady: Effect.die("A live server cannot admit installation"),
    setAutoDownload: () => Effect.die("Control cannot change preferences"),
  }
  return { calls, updates, state }
}
describe("running headless update control", () => {
  it("observes and prepares but rejects installation without closing the owner", async () => {
    const { calls, updates, state } = fixture()
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const control = yield* makeHeadlessUpdateControl(updates)
      expect((yield* control("status")).state).toEqual(state)
      expect(calls).toEqual([])
      const failure = yield* control("install").pipe(Effect.flip)
      expect(failure.message).toContain("Stop the server")
      expect(calls).toEqual([])
      for (const action of ["check", "download", "discard"] as const) {
        const reply = yield* control(action)
        yield* reply.afterReply
      }
      expect(calls).toEqual(["check", "download", "discard"])
    })).pipe(Effect.provide(TestContext.TestContext)))
  })
  it("owns one timer that ends with its scope", async () => {
    const { calls, updates } = fixture()
    await Effect.runPromise(Effect.gen(function* () {
      yield* Effect.scoped(Effect.gen(function* () {
        yield* makeHeadlessUpdateControl(updates)
        yield* TestClock.adjust("3 seconds")
        expect(calls).toEqual(["check"])
      }))
      yield* TestClock.adjust("2 hours")
      expect(calls).toEqual(["check"])
    }).pipe(Effect.provide(TestContext.TestContext)))
  })
})
