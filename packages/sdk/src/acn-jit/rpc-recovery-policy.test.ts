import { AcnRpc, AcnRpcRecoveryPolicyTag, AcnBoundary } from "@magnitudedev/acn-protocol"
import { Context, Option } from "effect"
import { describe, expect, it } from "vitest"
import { acnSubscriptionProtocol } from "./acn-subscription-protocol"

describe("ACN RPC recovery policy", () => {
  const recoveryPolicy = (tag: string) => {
    const operation = AcnRpc.operation(AcnBoundary, tag)
    if (operation === undefined) throw new TypeError(`Unknown ACN operation ${tag}`)
    return Option.getOrThrow(Context.getOption(operation.annotations, AcnRpcRecoveryPolicyTag))
  }

  it("classifies every finite RPC exactly once", () => {
    for (const { name: tag } of AcnRpc.operations(AcnBoundary)) {
      if (acnSubscriptionProtocol.isStream(tag)) continue
      expect(["ReplaySafe", "AtMostOnce"]).toContain(recoveryPolicy(tag))
    }
  })

  it("keeps side-effecting agent commands at-most-once", () => {
    expect(recoveryPolicy("SendMessage")).toBe("AtMostOnce")
    expect(recoveryPolicy("StartGoal")).toBe("AtMostOnce")
    expect(recoveryPolicy("RunBash")).toBe("AtMostOnce")
    expect(recoveryPolicy("UploadAttachment")).toBe("AtMostOnce")
  })
})
