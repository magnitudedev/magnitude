import type * as Rpc from "@effect/rpc/Rpc"
import { Schema } from "effect"
import { Operation } from "@magnitudedev/effect-query"
import * as RpcAdapter from "@magnitudedev/effect-query/rpc"
import {
  AcnRpcRecoveryPolicyTag,
  type AcnRpcRecoveryPolicy,
} from "../transport/recovery"

export interface AcnMutationPolicy {
  /** Commands must state whether retrying after an unknown outcome is safe. */
  readonly recovery: AcnRpcRecoveryPolicy
}

/** ACN policy interpretation and Effect RPC projection. It does not define operations. */
export const AcnRpc = RpcAdapter.make({
  decorate: (operation, rpc) => {
    const declaration = Operation.declaration(operation)
    if (declaration.kind === "subscription" || declaration.kind === "queryFromStream") return rpc

    const recovery: AcnRpcRecoveryPolicy = declaration.kind === "query"
      ? "ReplaySafe"
      : (declaration.policy as AcnMutationPolicy).recovery
    const finiteRpc = rpc as unknown as Rpc.Rpc<
      string,
      typeof Schema.Void,
      typeof Schema.Void,
      typeof Schema.Never
    >
    return finiteRpc.annotate(AcnRpcRecoveryPolicyTag, recovery)
  },
})
