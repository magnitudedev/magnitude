import { describe, expect, it } from "vitest"
import { Effect, Exit } from "effect"
import { ACN_SUBSCRIPTION_LIVENESS_TIMEOUT_MS } from "@magnitudedev/protocol"
import { acnSubscriptionProtocol, acnSubscriptionTags } from "./acn-subscription-protocol"

describe("ACN subscription protocol", () => {
  it("derives only explicitly declared subscription RPCs", () => {
    expect(acnSubscriptionTags).toEqual(new Set([
      "StreamActiveSessionStatuses",
      "StreamDisplayView",
      "WatchFile",
      "WatchMirroredStates",
    ]))
    expect(acnSubscriptionProtocol.isStream("CheckFileExists")).toBe(false)
  })

  it("consumes controls and forwards only encoded payload frames", async () => {
    const payload = { _tag: "payload" as const, payload: { event: "changed", path: "/x" } }
    const decoded = await Effect.runPromise(acnSubscriptionProtocol.decodeChunk([
      { _tag: "keepalive" },
      { _tag: "suspended", reason: "session-offloaded" },
      payload,
    ]))

    expect(decoded).toEqual({
      _tag: "Continue",
      values: [payload],
      progressed: true,
    })
  })

  it("reports authoritative termination without forwarding it", async () => {
    const decoded = await Effect.runPromise(acnSubscriptionProtocol.decodeChunk([
      { _tag: "terminated", reason: "acn-shutdown" },
    ]))
    expect(decoded).toEqual({ _tag: "Terminated" })
  })

  it("rejects malformed controls instead of guessing that they are payloads", async () => {
    const result = await Effect.runPromiseExit(
      acnSubscriptionProtocol.decodeChunk([{ _tag: "terminated", reason: "wrong" }]),
    )
    expect(Exit.isFailure(result)).toBe(true)
  })

  it("uses the protocol liveness deadline", () => {
    expect(acnSubscriptionProtocol.livenessTimeoutMs).toBe(
      ACN_SUBSCRIPTION_LIVENESS_TIMEOUT_MS,
    )
  })
})
