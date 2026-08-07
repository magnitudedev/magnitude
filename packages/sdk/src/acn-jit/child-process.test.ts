import { Effect, Fiber } from "effect"
import { describe, expect, it } from "vitest"
import { scopeAcnCandidate } from "./child-process"
import { AcnEnsuranceFailed } from "./errors"

const candidate = (options: {
  readonly releaseParentChannel: Effect.Effect<void, AcnEnsuranceFailed>
  readonly stopAndReap: Effect.Effect<void, AcnEnsuranceFailed>
}) =>
  scopeAcnCandidate({
    pid: 42,
    exited: Effect.never,
    ...options,
  })

describe("scopeAcnCandidate", () => {
  it("stops and reaps a candidate when its scope closes before admission", async () => {
    let stops = 0

    await Effect.runPromise(
      Effect.scoped(
        candidate({
          releaseParentChannel: Effect.void,
          stopAndReap: Effect.sync(() => {
            stops += 1
          }),
        }),
      ),
    )

    expect(stops).toBe(1)
  })

  it("disarms scoped cleanup only after successful admission observation", async () => {
    let releases = 0
    let stops = 0

    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const child = yield* candidate({
            releaseParentChannel: Effect.sync(() => {
              releases += 1
            }),
            stopAndReap: Effect.sync(() => {
              stops += 1
            }),
          })
          yield* child.admit
        }),
      ),
    )

    expect(releases).toBe(1)
    expect(stops).toBe(0)
  })

  it("keeps scoped cleanup armed when admission acknowledgement fails", async () => {
    let stops = 0

    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const child = yield* candidate({
            releaseParentChannel: Effect.fail(
              new AcnEnsuranceFailed({
                reason: "bootstrap pipe failed",
              }),
            ),
            stopAndReap: Effect.sync(() => {
              stops += 1
            }),
          })
          const result = yield* Effect.either(child.admit)
          expect(result._tag).toBe("Left")
        }),
      ),
    )

    expect(stops).toBe(1)
  })

  it("keeps cleanup armed when parent-channel release is interrupted", async () => {
    let stops = 0

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const child = yield* candidate({
        releaseParentChannel: Effect.never as Effect.Effect<void, AcnEnsuranceFailed>,
        stopAndReap: Effect.sync(() => {
          stops += 1
        }),
      })
      const admitting = yield* child.admit.pipe(Effect.fork)
      yield* Effect.yieldNow()
      yield* Fiber.interrupt(admitting)
    })))

    expect(stops).toBe(1)
  })

  it("permits only one admission acknowledgement", async () => {
    let releases = 0
    let stops = 0

    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const child = yield* candidate({
            releaseParentChannel: Effect.sync(() => {
              releases += 1
            }),
            stopAndReap: Effect.sync(() => {
              stops += 1
            }),
          })
          yield* child.admit
          const second = yield* Effect.either(child.admit)
          expect(second._tag).toBe("Left")
        }),
      ),
    )

    expect(releases).toBe(1)
    expect(stops).toBe(0)
  })
})
