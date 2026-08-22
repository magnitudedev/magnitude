import { Effect, Schema } from "effect"
import {
  AcnSubscriptionWireItem,
  ACN_SUBSCRIPTION_LIVENESS_TIMEOUT_MS,
  AcnRpc,
  AcnBoundary,
} from "@magnitudedev/acn-protocol"
import type { JsonValue } from "@magnitudedev/utils/schema"
import {
  isCleanOrInterruptedExit,
  type RecoveringStreamProtocol,
} from "../jit-rpc"

const decodeWireItem = Schema.decodeUnknown(AcnSubscriptionWireItem)

/**
 * Consumes ACN subscription controls and unwraps payload frames so Effect RPC
 * only ever sees encoded domain values. Every stream Rpc of the ACN group is a
 * subscription; framing is decided by the group, not by annotations.
 */
const decodeChunk: RecoveringStreamProtocol["decodeChunk"] = (values) =>
  Effect.gen(function* () {
    const decoded = yield* Effect.forEach(values, (value) => decodeWireItem(value))
    const payloads: JsonValue[] = []
    let terminated = false

    for (const item of decoded) {
      switch (item._tag) {
        case "payload":
          payloads.push(item.payload)
          break
        case "keepalive":
          break
        case "terminated":
          terminated = true
          break
      }
    }

    return terminated
      ? { _tag: "Terminated" }
      : { _tag: "Continue", values: payloads, progressed: payloads.length > 0 }
  })

export const acnSubscriptionProtocol: RecoveringStreamProtocol = {
  isStream: (tag) => AcnRpc.operation(AcnBoundary, tag)?.stream === true,
  decodeChunk,
  livenessTimeoutMs: ACN_SUBSCRIPTION_LIVENESS_TIMEOUT_MS,
  isExitWithoutTermination: isCleanOrInterruptedExit,
}
