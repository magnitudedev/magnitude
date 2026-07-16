import { afterEach, describe, expect, test } from "vitest"
import { Effect } from "effect"
import {
  clearEphemeralMessage,
  getEphemeralMessageSnapshot,
} from "@magnitudedev/client-common"
import { makeCliEffectLoggingLayer } from "./effect-logger"

afterEach(() => clearEphemeralMessage())

const runLog = (debug: boolean, effect: Effect.Effect<void>): Promise<void> =>
  Effect.runPromise(effect.pipe(
    Effect.provide(makeCliEffectLoggingLayer({ debug })),
  ))

describe("CLI Effect logger", () => {
  test("surfaces errors as toasts", async () => {
    await runLog(false, Effect.logError("RPC failed"))

    expect(getEphemeralMessageSnapshot()).toMatchObject({
      text: "RPC failed",
      tone: "error",
    })
  })

  test("does not surface warnings normally", async () => {
    await runLog(false, Effect.logWarning("retrying"))

    expect(getEphemeralMessageSnapshot()).toBeNull()
  })

  test("surfaces warnings in debug mode", async () => {
    await runLog(true, Effect.logWarning("retrying"))

    expect(getEphemeralMessageSnapshot()).toMatchObject({
      text: "retrying",
      tone: "warning",
    })
  })
})
