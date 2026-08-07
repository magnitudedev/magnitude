import { Context, Schema } from "effect"

export const AcnRpcRecoveryPolicySchema = Schema.Literal("ReplaySafe", "AtMostOnce")
export type AcnRpcRecoveryPolicy = typeof AcnRpcRecoveryPolicySchema.Type

export class AcnRpcRecoveryPolicyTag extends Context.Tag("AcnRpcRecoveryPolicy")<
  AcnRpcRecoveryPolicyTag,
  AcnRpcRecoveryPolicy
>() {}
