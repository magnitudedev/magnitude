import { describe, expect, it } from "vitest"
import { Effect, Ref, Stream } from "effect"
import { propagateModelConfigurationChanges } from "./model-configuration-propagation"

describe("model configuration propagation", () => {
  it("refreshes stable provider clients before every active session", async () => {
    const events = await Effect.runPromise(Effect.gen(function* () {
      const observed = yield* Ref.make<readonly string[]>([])
      const record = (event: string) => Ref.update(observed, (current) => [...current, event])

      yield* propagateModelConfigurationChanges(
        Stream.make(undefined),
        record("providers"),
        Effect.succeed([
          { session: { refreshConfig: () => record("session-a") } },
          { session: { refreshConfig: () => record("session-b") } },
        ]),
      )

      return yield* Ref.get(observed)
    }))

    expect(events[0]).toBe("providers")
    expect(new Set(events.slice(1))).toEqual(new Set(["session-a", "session-b"]))
  })
})
