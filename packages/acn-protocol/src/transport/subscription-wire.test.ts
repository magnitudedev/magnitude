import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import { AcnSubscriptionPayloadFrame, AcnSubscriptionWireItem } from "./subscription-wire"

describe("ACN subscription wire frames", () => {
  it("decodes payload frames and controls", () => {
    expect(Schema.decodeUnknownSync(AcnSubscriptionWireItem)({
      _tag: "payload",
      payload: { value: "hello" },
    })).toEqual({ _tag: "payload", payload: { value: "hello" } })
    expect(Schema.decodeUnknownSync(AcnSubscriptionWireItem)({ _tag: "keepalive" })).toEqual({
      _tag: "keepalive",
    })
    expect(Schema.decodeUnknownSync(AcnSubscriptionWireItem)({
      _tag: "terminated",
      reason: "acn-shutdown",
    })).toEqual({ _tag: "terminated", reason: "acn-shutdown" })
  })

  it("rejects malformed controls", () => {
    expect(() => Schema.decodeUnknownSync(AcnSubscriptionWireItem)({
      _tag: "terminated",
      reason: "wrong",
    })).toThrow()
  })

  it("encodes a payload frame as a plain object", () => {
    expect(Schema.encodeSync(AcnSubscriptionPayloadFrame)({ _tag: "payload", payload: 1 })).toEqual({
      _tag: "payload",
      payload: 1,
    })
  })
})
