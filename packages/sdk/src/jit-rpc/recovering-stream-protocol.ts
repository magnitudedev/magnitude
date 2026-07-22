import type { ResponseExitEncoded } from "@effect/rpc/RpcMessage"
import type { JsonValue } from "@magnitudedev/utils/schema"
import type { Effect, ParseResult, Schema } from "effect"

export type DecodedStreamChunk =
  | {
      readonly _tag: "Continue"
      /** Encoded domain values that remain after protocol controls are consumed. */
      readonly values: ReadonlyArray<JsonValue>
      readonly progressed: boolean
    }
  | { readonly _tag: "Terminated" }

/**
 * Extension point for a daemon-specific stream wire protocol. Generic JIT RPC
 * recovery owns retries and endpoint generations; it never interprets the
 * daemon's control values itself.
 */
export interface RecoveringStreamProtocol {
  readonly isStream: (rpcTag: string) => boolean
  readonly decodeChunk: (
    values: ReadonlyArray<JsonValue>,
  ) => Effect.Effect<DecodedStreamChunk, ParseResult.ParseError>
  readonly livenessTimeoutMs: number
  /** Recognizes stream exits that violate the required terminal-control handshake. */
  readonly isExitWithoutTermination: (exit: ResponseExitEncoded["exit"]) => boolean
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

export const isInterruptedExit = (exit: ResponseExitEncoded["exit"]): boolean =>
  exit._tag === "Failure" && isInterruptOnlyCause(exit.cause)
