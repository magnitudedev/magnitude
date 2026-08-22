import { Rpc } from "@effect/rpc"
import { Context, type Schedule, Schema } from "effect"
import { Acn } from "../boundary"
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

/**
 * An ACN subscription defined through the contract: a core Effect Query
 * subscription whose Rpc carries the ACN subscription wire protocol.
 */
export const acnSubscription = <
  const Tag extends string,
  PayloadType,
  PayloadEncoded,
  SuccessType,
  SuccessEncoded,
  SuccessRequirements,
  Error extends Schema.Schema.All,
>(
  tag: Tag,
  options: {
    readonly payload: Schema.Schema<PayloadType, PayloadEncoded, never>
    readonly success: Schema.Schema<SuccessType, SuccessEncoded, SuccessRequirements>
    readonly error: Error
    readonly scope?: "global" | "session"
    readonly reconnect?: Schedule.Schedule<unknown, unknown, never>
  },
) =>
  Acn.subscription(tag, {
    payload: options.payload,
    success: AcnSubscriptionPayload(options.success),
    error: options.error,
    annotations: Context.make(AcnSubscriptionMetadataTag, { scope: options.scope ?? "global" }),
    reconnect: options.reconnect,
  })
