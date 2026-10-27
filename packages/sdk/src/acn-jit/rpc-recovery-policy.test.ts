import { AcnRpcRecoveryPolicyTag, AcnRpcGroup } from "@magnitudedev/acn-protocol"
import { Context, Option } from "effect"
import { describe, expect, it } from "vitest"
import { acnSubscriptionProtocol } from "./acn-subscription-protocol"

describe("ACN RPC recovery policy", () => {
  const recoveryPolicy = (tag: string) => {
    const operation = AcnRpcGroup.requests.get(tag)
    if (operation === undefined) throw new TypeError(`Unknown ACN operation ${tag}`)
    return Option.getOrThrow(Context.getOption(operation.annotations, AcnRpcRecoveryPolicyTag))
  }

  it("classifies every finite RPC exactly once", () => {
    for (const { _tag: tag } of [...AcnRpcGroup.requests.values()]) {
      if (acnSubscriptionProtocol.isStream(tag)) continue
      expect(["ReplaySafe", "AtMostOnce"]).toContain(recoveryPolicy(tag))
    }
  })

  it("replays a chat's start, which its IDs make idempotent", () => {
    expect(recoveryPolicy("StartChat")).toBe("ReplaySafe")
  })
})
