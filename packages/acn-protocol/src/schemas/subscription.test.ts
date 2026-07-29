import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import { AcnSubscriptionPayload } from "./subscription"

describe("AcnSubscriptionPayload", () => {
  const Payload = Schema.Struct({ value: Schema.String })
  const SubscriptionPayload = AcnSubscriptionPayload(Payload)

  it("encodes a domain payload as a payload frame", () => {
    expect(Schema.encodeSync(SubscriptionPayload)({ value: "hello" })).toEqual({
      _tag: "payload",
      payload: { value: "hello" },
    })
  })

  it("decodes a payload frame back to the domain payload", () => {
    expect(Schema.decodeUnknownSync(SubscriptionPayload)({
      _tag: "payload",
      payload: { value: "hello" },
    })).toEqual({ value: "hello" })
  })
})
