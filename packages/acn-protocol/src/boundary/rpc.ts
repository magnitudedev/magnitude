import type * as Rpc from "@effect/rpc/Rpc"
import { Schema } from "effect"
import { Operation } from "@magnitudedev/effect-query"
import * as RpcAdapter from "@magnitudedev/effect-query/rpc"
import { AcnRpcDemand, AcnRpcDemandPolicyTag } from "../transport/demand"
import {
  AcnRpcRecoveryPolicyTag,
  type AcnRpcRecoveryPolicy,
} from "../transport/recovery"

export interface AcnQueryPolicy {
  /** Finite observations hold residency unless explicitly lifecycle-neutral. */
  readonly demand?: boolean
}

export interface AcnMutationPolicy {
  /** Commands must state whether retrying after an unknown outcome is safe. */
  readonly recovery: AcnRpcRecoveryPolicy
  /** Finite commands hold residency unless explicitly lifecycle-neutral. */
  readonly demand?: boolean
}

/** ACN policy interpretation and Effect RPC projection. It does not define operations. */
export const AcnRpc = RpcAdapter.make<typeof AcnRpcDemand>({
  decorate: (operation, rpc) => {
    const declaration = Operation.declaration(operation)
    if (declaration.kind === "subscription" || declaration.kind === "queryFromStream") return rpc

    const policy = declaration.policy as AcnQueryPolicy | AcnMutationPolicy
    const recovery: AcnRpcRecoveryPolicy = declaration.kind === "query"
      ? "ReplaySafe"
      : (policy as AcnMutationPolicy).recovery
    const demand = policy.demand !== false
    const finiteRpc = rpc as unknown as Rpc.Rpc<
      string,
      typeof Schema.Void,
      typeof Schema.Void,
      typeof Schema.Never
    >
    return finiteRpc
      .annotate(AcnRpcRecoveryPolicyTag, recovery)
      .annotate(AcnRpcDemandPolicyTag, demand)
      .middleware(AcnRpcDemand)
  },
})
