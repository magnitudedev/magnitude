import { Context, Effect, Option, Schema } from "effect"
import {
  AcnSubscriptionMetadataTag,
  AcnSubscriptionWireItem,
  MagnitudeRpcs,
  ACN_SUBSCRIPTION_LIVENESS_TIMEOUT_MS,
} from "@magnitudedev/protocol"
import {
  isCleanOrInterruptedExit,
  type RecoveringStreamProtocol,
} from "../jit-rpc"

export const acnSubscriptionTags: ReadonlySet<string> = new Set(
  Array.from(MagnitudeRpcs.requests.entries())
    .filter(([, rpc]) =>
      Option.isSome(Context.getOption(rpc.annotations, AcnSubscriptionMetadataTag)),
    )
    .map(([tag]) => tag),
)

const decodeWireItem = Schema.decodeUnknown(AcnSubscriptionWireItem)

const decodeChunk: RecoveringStreamProtocol["decodeChunk"] = (values) =>
  Effect.gen(function* () {
    const decoded = yield* Effect.forEach(values, (value) => decodeWireItem(value))
    const payloads: Extract<AcnSubscriptionWireItem, { readonly _tag: "payload" }>[] = []
    let progressed = false
    let terminated = false

    for (const item of decoded) {
      switch (item._tag) {
        case "payload":
          payloads.push(item)
          progressed = true
          break
        case "keepalive":
          break
        case "suspended":
          progressed = true
          break
        case "terminated":
          terminated = true
          break
      }
    }

    return terminated
      ? { _tag: "Terminated" }
      : { _tag: "Continue", values: payloads, progressed }
  })

export const acnSubscriptionProtocol: RecoveringStreamProtocol = {
  isStream: (rpcTag) => acnSubscriptionTags.has(rpcTag),
  decodeChunk,
  livenessTimeoutMs: ACN_SUBSCRIPTION_LIVENESS_TIMEOUT_MS,
  isExitWithoutTermination: isCleanOrInterruptedExit,
}
