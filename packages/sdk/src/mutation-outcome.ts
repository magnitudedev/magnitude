import { RpcClientError } from "@effect/rpc"
import { RpcOutcomeUnknown } from "./jit-rpc/errors"

/**
 * Whether an at-most-once RPC lost its acknowledgement after it may have
 * reached the daemon. Callers must reconcile these failures from
 * authoritative state instead of treating them as domain rejection.
 */
export function isRpcOutcomeUnknown(error: unknown): boolean {
  return error instanceof RpcClientError.RpcClientError &&
    error.cause instanceof RpcOutcomeUnknown
}
