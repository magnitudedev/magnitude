import { Rpc } from "@effect/rpc"
import { Context, Schema } from "effect"
import { AcnSubscriptionPayload } from "../schemas/subscription"

export interface AcnSubscriptionMetadata {
  /** Session scope allows the ACN to suspend only affected display subscriptions. */
  readonly scope: "global" | "session"
}

export class AcnSubscriptionMetadataTag extends Context.Tag("AcnSubscriptionMetadata")<
  AcnSubscriptionMetadataTag,
  AcnSubscriptionMetadata
>() {}

/**
 * Defines a stream whose domain payload is carried by the ACN subscription
 * wire protocol. The returned RPC remains typed as `Stream<Payload>`.
 */
export const makeAcnSubscriptionRpc = <
  const Tag extends string,
  PayloadType,
  PayloadEncoded,
  PayloadRequirements,
  SuccessType,
  SuccessEncoded,
  SuccessRequirements,
  Error extends Schema.Schema.All,
>(
  tag: Tag,
  options: {
    readonly payload: Schema.Schema<PayloadType, PayloadEncoded, PayloadRequirements>
    readonly success: Schema.Schema<SuccessType, SuccessEncoded, SuccessRequirements>
    readonly error: Error
    readonly scope?: "global" | "session"
  },
) =>
  Rpc.make(tag, {
    payload: options.payload,
    success: AcnSubscriptionPayload(options.success),
    error: options.error,
    stream: true,
  }).annotate(AcnSubscriptionMetadataTag, { scope: options.scope ?? "global" })
