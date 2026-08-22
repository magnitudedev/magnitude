import { Schema } from "effect"
import { JsonValueSchema } from "@magnitudedev/utils/schema"

/**
 * ACN subscription wire protocol values.
 *
 * These schemas describe frames on the wire only. Subscription definitions
 * carry domain schemas; the ACN protocol decorator wraps each encoded domain
 * value in a `payload` frame and interleaves controls, and the SDK protocol
 * decorator consumes controls and unwraps payloads before Effect RPC decodes
 * them. Neither handlers nor consumers observe frames.
 */

/** Cadence at which the ACN emits keepalive controls on open subscriptions. */
export const ACN_SUBSCRIPTION_KEEPALIVE_INTERVAL_MS = 5_000
/** Three missed keepalives classify the subscription transport as dead. */
export const ACN_SUBSCRIPTION_LIVENESS_TIMEOUT_MS = 15_000

export const AcnSubscriptionKeepalive = Schema.TaggedStruct("keepalive", {})
export type AcnSubscriptionKeepalive = typeof AcnSubscriptionKeepalive.Type

export const AcnSubscriptionTerminated = Schema.TaggedStruct("terminated", {
  reason: Schema.Literal("acn-shutdown"),
})
export type AcnSubscriptionTerminated = typeof AcnSubscriptionTerminated.Type

/** Control values belonging to the ACN subscription wire protocol. */
export const AcnSubscriptionControl = Schema.Union(
  AcnSubscriptionKeepalive,
  AcnSubscriptionTerminated,
)
export type AcnSubscriptionControl = typeof AcnSubscriptionControl.Type

/** The frame carrying one encoded domain value. */
export const AcnSubscriptionPayloadFrame = Schema.TaggedStruct("payload", { payload: JsonValueSchema })
export type AcnSubscriptionPayloadFrame = typeof AcnSubscriptionPayloadFrame.Type

/** Strict codec used only at the encoded transport boundary. */
export const AcnSubscriptionWireItem = Schema.Union(
  AcnSubscriptionPayloadFrame,
  AcnSubscriptionControl,
)
export type AcnSubscriptionWireItem = typeof AcnSubscriptionWireItem.Type
