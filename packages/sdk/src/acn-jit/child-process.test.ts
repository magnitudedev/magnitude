import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { scopePreHandoffCandidate } from "./child-process"
import { DaemonSpawnFailed } from "./errors"

const candidate = (options: {
  readonly releaseForHandoff: Effect.Effect<void, DaemonSpawnFailed>
  readonly stopAndReap: Effect.Effect<void, DaemonSpawnFailed>
}) =>
  scopePreHandoffCandidate({
    pid: 42,
    ...options,
  })

describe("scopePreHandoffCandidate", () => {
  it("stops and reaps a candidate when its scope closes before handoff", async () => {
    let stops = 0

    await Effect.runPromise(
      Effect.scoped(
        candidate({
          releaseForHandoff: Effect.void,
          stopAndReap: Effect.sync(() => {
            stops += 1
          }),
        }),
      ),
    )

    expect(stops).toBe(1)
  })

  it("disarms scoped cleanup only after successful handoff", async () => {
    let releases = 0
    let stops = 0

    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const child = yield* candidate({
            releaseForHandoff: Effect.sync(() => {
              releases += 1
            }),
            stopAndReap: Effect.sync(() => {
              stops += 1
            }),
          })
          yield* child.handoff
        }),
      ),
    )

    expect(releases).toBe(1)
    expect(stops).toBe(0)
  })

  it("keeps scoped cleanup armed when handoff fails", async () => {
    let stops = 0

    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const child = yield* candidate({
            releaseForHandoff: Effect.fail(
              new DaemonSpawnFailed({
                reason: "bootstrap pipe failed",
              }),
            ),
            stopAndReap: Effect.sync(() => {
              stops += 1
            }),
          })
          const result = yield* Effect.either(child.handoff)
          expect(result._tag).toBe("Left")
        }),
      ),
    )

    expect(stops).toBe(1)
  })

  it("permits only one handoff attempt", async () => {
    let releases = 0
    let stops = 0

    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const child = yield* candidate({
            releaseForHandoff: Effect.sync(() => {
              releases += 1
            }),
            stopAndReap: Effect.sync(() => {
              stops += 1
            }),
          })
          yield* child.handoff
          const second = yield* Effect.either(child.handoff)
          expect(second._tag).toBe("Left")
        }),
      ),
    )

    expect(releases).toBe(1)
    expect(stops).toBe(0)
  })
})
