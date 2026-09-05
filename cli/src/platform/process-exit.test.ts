import { Deferred, Effect, Exit, Ref } from "effect"
import { describe, expect, it } from "vitest"
import { untilProcessExit, type ProcessExitRequest } from "./process-exit"

const source = () => Effect.map(Deferred.make<ProcessExitRequest>(), (request) => ({
  request,
  exit: { await: Deferred.await(request) },
}))

describe("untilProcessExit", () => {
  it("propagates a startup failure immediately instead of waiting for an exit signal", async () => {
    const started = Date.now()
    const exit = await Effect.runPromiseExit(Effect.gen(function* () {
      const { exit } = yield* source()
      return yield* untilProcessExit(Effect.fail("service failed to start"), exit)
    }))
    expect(Exit.isFailure(exit)).toBe(true)
    expect(Date.now() - started).toBeLessThan(1_000)
  })

  it("interrupts startup on an exit request and runs its finalizers", async () => {
    const outcome = await Effect.runPromise(Effect.gen(function* () {
      const { request, exit } = yield* source()
      const released = yield* Ref.make(false)
      const startup = Effect.acquireRelease(Effect.void, () => Ref.set(released, true)).pipe(
        Effect.zipRight(Effect.never),
        Effect.scoped,
      )
      const result = yield* Effect.fork(untilProcessExit(startup, exit))
      yield* Deferred.succeed(request, { _tag: "Signal", signal: "SIGINT" })
      return { result: yield* result.await, released: yield* Ref.get(released) }
    }))
    expect(Exit.isSuccess(outcome.result) && outcome.result.value._tag).toBe("Exit")
    expect(outcome.released).toBe(true)
  })

  it("returns the startup value when it completes first", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const { exit } = yield* source()
      return yield* untilProcessExit(Effect.succeed(42), exit)
    }))
    expect(result).toEqual({ _tag: "Completed", value: 42 })
  })
})
