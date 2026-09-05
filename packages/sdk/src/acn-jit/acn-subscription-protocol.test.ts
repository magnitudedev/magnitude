import { describe, expect, it } from "vitest"
import { Effect, Exit } from "effect"
import { ACN_SUBSCRIPTION_LIVENESS_TIMEOUT_MS, AcnRpcGroup } from "@magnitudedev/acn-protocol"
import { acnSubscriptionProtocol } from "./acn-subscription-protocol"

describe("ACN subscription protocol", () => {
  it("treats exactly the group's stream Rpcs as subscriptions", () => {
    const subscriptions = [...AcnRpcGroup.requests.values()]
      .map((operation) => operation._tag)
      .filter(acnSubscriptionProtocol.isStream)
      .sort()
    expect(subscriptions).toEqual([
      "StreamActiveSessionStatuses",
      "StreamChanges",
      "StreamDisplayView",
      "WatchFile",
      "WatchProjectFiles",
    ])
    expect(acnSubscriptionProtocol.isStream("CheckFileExists")).toBe(false)
  })

  it("consumes controls and unwraps payload frames to encoded domain values", async () => {
    const decoded = await Effect.runPromise(acnSubscriptionProtocol.decodeChunk([
      { _tag: "keepalive" },
      { _tag: "payload", payload: { event: "changed", path: "/x" } },
      { _tag: "payload", payload: 42 },
    ]))

    expect(decoded).toEqual({
      _tag: "Continue",
      values: [{ event: "changed", path: "/x" }, 42],
      progressed: true,
    })
  })

  it("reports keepalive-only chunks as no progress", async () => {
    const decoded = await Effect.runPromise(acnSubscriptionProtocol.decodeChunk([{ _tag: "keepalive" }]))
    expect(decoded).toEqual({ _tag: "Continue", values: [], progressed: false })
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
