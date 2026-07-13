import { RpcSchema } from "@effect/rpc"
import { Option } from "effect"
import {
  MagnitudeRpcs,
  STREAM_LIVENESS_TIMEOUT_MS,
} from "@magnitudedev/protocol"
import {
  isCleanOrInterruptedExit,
  type ResidentStreamPolicy,
} from "../jit-rpc"

export const acnResidentStreamTags: ReadonlySet<string> = new Set(
  Array.from(MagnitudeRpcs.requests.entries())
    .filter(([, rpc]) => Option.isSome(RpcSchema.getStreamSchemas(rpc.successSchema.ast)))
    .map(([tag]) => tag),
)

export const isEncodedHeartbeat = (value: unknown): boolean =>
  typeof value === "object" && value !== null && "_tag" in value && value._tag === "heartbeat"

export const acnResidentStreamPolicy: ResidentStreamPolicy = {
  isResident: (rpcTag) => acnResidentStreamTags.has(rpcTag),
  isHeartbeatChunk: isEncodedHeartbeat,
  livenessTimeoutMs: STREAM_LIVENESS_TIMEOUT_MS,
  isRelinquishExit: isCleanOrInterruptedExit,
}
