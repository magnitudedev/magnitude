import { describe, expect, it } from "bun:test"
import { Effect } from "effect"

import type { BaseEvent, Timestamped } from "./event-bus-core"
import { EventSinkTag, makeEventSinkLayer } from "./event-sink"

interface TestEvent extends BaseEvent {
  readonly type: "test"
  readonly value: string
}

const event = (value: string, timestamp: number): Timestamped<TestEvent> => ({
  type: "test",
  value,
  timestamp,
})

describe("EventSink", () => {
  it("acknowledges the exact claimed prefix while retaining later events", async () => {
    const remaining = await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const sink = yield* EventSinkTag<TestEvent>()
          yield* sink.append(event("a", 1))
          yield* sink.append(event("b", 2))

          yield* Effect.scoped(
            Effect.gen(function* () {
              const claim = yield* sink.claimPending()
              yield* sink.append(event("c", 3))
              yield* claim.acknowledge
            })
          )

          const nextClaim = yield* sink.claimPending()
          return nextClaim.events
        })
      ).pipe(Effect.provide(makeEventSinkLayer<TestEvent>()))
    )

    expect(remaining).toEqual([event("c", 3)])
  })

  it("leaves a claim pending when its scope closes without acknowledgement", async () => {
    const pending = await Effect.runPromise(
      Effect.gen(function* () {
        const sink = yield* EventSinkTag<TestEvent>()
        yield* sink.append(event("a", 1))

        yield* Effect.scoped(sink.claimPending())

        return yield* Effect.scoped(
          sink.claimPending().pipe(Effect.map(claim => claim.events))
        )
      }).pipe(Effect.provide(makeEventSinkLayer<TestEvent>()))
    )

    expect(pending).toEqual([event("a", 1)])
  })

  it("defects instead of acknowledging the same claim twice", async () => {
    const result = await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const sink = yield* EventSinkTag<TestEvent>()
          yield* sink.append(event("a", 1))
          const claim = yield* sink.claimPending()
          yield* claim.acknowledge
          return yield* Effect.exit(claim.acknowledge)
        })
      ).pipe(Effect.provide(makeEventSinkLayer<TestEvent>()))
    )

    expect(result._tag).toBe("Failure")
  })

  it("defects instead of acknowledging a claim after its scope closes", async () => {
    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const sink = yield* EventSinkTag<TestEvent>()
        yield* sink.append(event("a", 1))
        const claim = yield* Effect.scoped(sink.claimPending())
        const exit = yield* Effect.exit(claim.acknowledge)
        const pending = yield* Effect.scoped(
          sink.claimPending().pipe(Effect.map(nextClaim => nextClaim.events))
        )
        return { exit, pending }
      }).pipe(Effect.provide(makeEventSinkLayer<TestEvent>()))
    )

    expect(result.exit._tag).toBe("Failure")
    expect(result.pending).toEqual([event("a", 1)])
  })
})
