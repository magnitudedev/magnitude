import { describe, expect, it } from "vitest"
import { RpcClientError } from "@effect/rpc"
import { RpcOutcomeUnknown, TransportRequestFailed } from "./jit-rpc/errors"
import { isRpcOutcomeUnknown } from "./mutation-outcome"

describe("isRpcOutcomeUnknown", () => {
  it("recognizes the typed at-most-once ambiguity", () => {
    expect(isRpcOutcomeUnknown(new RpcClientError.RpcClientError({
      reason: "Unknown",
      message: "RpcOutcomeUnknown",
      cause: new RpcOutcomeUnknown({ tag: "SendMessage" }),
    }))).toBe(true)
  })

  it("does not classify ordinary transport or domain failures as ambiguous", () => {
    expect(isRpcOutcomeUnknown(new RpcClientError.RpcClientError({
      reason: "Unknown",
      message: "TransportRequestFailed",
      cause: new TransportRequestFailed({ message: "refused" }),
    }))).toBe(false)
    expect(isRpcOutcomeUnknown(new Error("rejected"))).toBe(false)
  })
})
