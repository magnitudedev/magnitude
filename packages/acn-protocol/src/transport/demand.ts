import { RpcMiddleware } from "@effect/rpc"
import { Context } from "effect"

/** Stable metadata describing whether an operation holds ACN residency. */
export class AcnRpcDemandPolicyTag extends Context.Tag("AcnRpcDemandPolicy")<
  AcnRpcDemandPolicyTag,
  boolean
>() {}

/** Holds ACN residency for the complete lifetime of one finite operation. */
export class AcnRpcDemand extends RpcMiddleware.Tag<AcnRpcDemand>()("AcnRpcDemand", {
  wrap: true,
}) {}
