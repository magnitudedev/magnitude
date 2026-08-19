import { Effect, Exit, Scope } from "effect"
import { describe, expect, it } from "vitest"
import { makeProcessExitSource } from "./process-exit"

describe("process exit source", () => {
  it("turns signals into data and removes every listener with its scope", async () => {
    const signals = [
      "SIGINT",
      "SIGTERM",
      "SIGHUP",
      "beforeExit",
      "exit",
      "uncaughtException",
      "unhandledRejection",
    ] as const
    const initialCounts = signals.map((signal) => process.listenerCount(signal))
    const initialSigintListeners = new Set(process.rawListeners("SIGINT"))
    const scope = await Effect.runPromise(Scope.make())

    try {
      const source = await Effect.runPromise(makeProcessExitSource.pipe(
        Effect.provideService(Scope.Scope, scope),
      ))
      expect(signals.map((signal) => process.listenerCount(signal)))
        .toEqual(initialCounts.map((count) => count + 1))

      const signalListener = process.rawListeners("SIGINT")
        .find((listener) => !initialSigintListeners.has(listener))
      expect(signalListener).toBeDefined()
      signalListener!()
      await expect(Effect.runPromise(source.await)).resolves.toEqual({
        _tag: "Signal",
        signal: "SIGINT",
      })
    } finally {
      await Effect.runPromise(Scope.close(scope, Exit.void))
    }

    expect(signals.map((signal) => process.listenerCount(signal)))
      .toEqual(initialCounts)
  })
})
