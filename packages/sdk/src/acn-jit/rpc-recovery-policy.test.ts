import { MagnitudeRpcs, acnRpcRecoveryPolicy } from "@magnitudedev/acn-protocol"
import { describe, expect, it } from "vitest"
import { acnSubscriptionProtocol } from "./acn-subscription-protocol"

describe("ACN RPC recovery policy", () => {
  it("classifies every finite RPC exactly once", () => {
    for (const tag of MagnitudeRpcs.requests.keys()) {
      if (acnSubscriptionProtocol.isStream(tag)) continue
      expect(["ReplaySafe", "AtMostOnce"]).toContain(acnRpcRecoveryPolicy(tag))
    }
  })

  it("keeps side-effecting agent commands at-most-once", () => {
    expect(acnRpcRecoveryPolicy("SendMessage")).toBe("AtMostOnce")
    expect(acnRpcRecoveryPolicy("StartGoal")).toBe("AtMostOnce")
    expect(acnRpcRecoveryPolicy("RunBash")).toBe("AtMostOnce")
    expect(acnRpcRecoveryPolicy("UploadAttachment")).toBe("AtMostOnce")
  })
})
