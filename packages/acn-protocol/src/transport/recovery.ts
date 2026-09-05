import { Rpc, RpcSchema } from "@effect/rpc"
import type { RpcMiddleware } from "@effect/rpc"
import { Brand, Context, Schema } from "effect"

export const AcnRpcRecoveryPolicySchema = Schema.Literal("ReplaySafe", "AtMostOnce")
export type AcnRpcRecoveryPolicy = typeof AcnRpcRecoveryPolicySchema.Type

export class AcnRpcRecoveryPolicyTag extends Context.Tag("AcnRpcRecoveryPolicy")<
  AcnRpcRecoveryPolicyTag,
  AcnRpcRecoveryPolicy
>() {}

export type RecoveryDeclared = Brand.Brand<"RpcRecoveryDeclared">

/** Finite requests must explicitly choose whether replay is safe. This is not cache policy. */
const withRecovery =
  (policy: AcnRpcRecoveryPolicy) =>
  <Tag extends string, Payload extends Schema.Schema.All, Success extends Schema.Schema.Any, Error extends Schema.Schema.All, Middleware extends RpcMiddleware.TagClassAny>(
    rpc: Rpc.Rpc<Tag, Payload, Success, Error, Middleware>
  ) => {
    if (RpcSchema.isStreamSchema(rpc.successSchema))
      throw new TypeError("Recovery policy applies only to finite RPCs")
    const declared = Brand.nominal<Rpc.Rpc<Tag, Payload, Success, Error, Middleware> & RecoveryDeclared>()
    return declared(rpc.annotate(AcnRpcRecoveryPolicyTag, policy))
  }

export const replaySafe = withRecovery("ReplaySafe")
export const atMostOnce = withRecovery("AtMostOnce")
