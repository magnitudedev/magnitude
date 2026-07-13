import type { ResponseExitEncoded } from "@effect/rpc/RpcMessage"
import type { Schema } from "effect"

export interface ResidentStreamPolicy {
  readonly isResident: (rpcTag: string) => boolean
  readonly isHeartbeatChunk: (value: unknown) => boolean
  readonly livenessTimeoutMs: number
  readonly isRelinquishExit: (exit: ResponseExitEncoded["exit"]) => boolean
}

export const isInterruptOnlyCause = (cause: Schema.CauseEncoded<unknown, unknown>): boolean => {
  switch (cause._tag) {
    case "Empty":
    case "Interrupt":
      return true
    case "Sequential":
    case "Parallel":
      return isInterruptOnlyCause(cause.left) && isInterruptOnlyCause(cause.right)
    case "Fail":
    case "Die":
      return false
  }
}

export const isCleanOrInterruptedExit = (exit: ResponseExitEncoded["exit"]): boolean =>
  exit._tag === "Success" || isInterruptOnlyCause(exit.cause)
