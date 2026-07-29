import { Schema } from "effect"
import { JsonValueSchema } from "@magnitudedev/utils/schema"

/** Cadence at which the ACN emits keepalive controls on open subscriptions. */
export const ACN_SUBSCRIPTION_KEEPALIVE_INTERVAL_MS = 5_000
/** Three missed keepalives classify the subscription transport as dead. */
export const ACN_SUBSCRIPTION_LIVENESS_TIMEOUT_MS = 15_000

export const AcnSubscriptionKeepalive = Schema.TaggedStruct("keepalive", {})
export type AcnSubscriptionKeepalive = typeof AcnSubscriptionKeepalive.Type

export const AcnSubscriptionSuspended = Schema.TaggedStruct("suspended", {
  reason: Schema.Literal("session-offloaded"),
})
export type AcnSubscriptionSuspended = typeof AcnSubscriptionSuspended.Type

export const AcnSubscriptionTerminated = Schema.TaggedStruct("terminated", {
  reason: Schema.Literal("acn-shutdown"),
})
export type AcnSubscriptionTerminated = typeof AcnSubscriptionTerminated.Type

/** Control values belonging to the ACN subscription wire protocol. */
export const AcnSubscriptionControl = Schema.Union(
  AcnSubscriptionKeepalive,
  AcnSubscriptionSuspended,
  AcnSubscriptionTerminated,
)
export type AcnSubscriptionControl = typeof AcnSubscriptionControl.Type

export const AcnSubscriptionPayloadFrame = <Payload, Encoded, Requirements>(
  payload: Schema.Schema<Payload, Encoded, Requirements>,
) => Schema.TaggedStruct("payload", { payload })

/**
 * Schema used by subscription RPCs.
 *
 * Its decoded type is the domain payload. Its encoded representation is the
 * subscription protocol's payload frame. Consequently ACN handlers produce
 * `Stream<Payload>` and SDK consumers receive `Stream<Payload>`; framing is a
 * transport concern and cannot leak into either API.
 */
export const AcnSubscriptionPayload = <Payload, Encoded, Requirements>(
  payload: Schema.Schema<Payload, Encoded, Requirements>,
) =>
  Schema.transform(AcnSubscriptionPayloadFrame(payload), payload, {
    strict: true,
    decode: (_frame, encodedFrame) => encodedFrame.payload,
    encode: (_encodedValue, value) => ({ _tag: "payload" as const, payload: value }),
  })

/** Strict codec used only at the encoded transport boundary. */
export const AcnSubscriptionWireItem = Schema.Union(
  AcnSubscriptionPayloadFrame(JsonValueSchema),
  AcnSubscriptionControl,
)
export type AcnSubscriptionWireItem = typeof AcnSubscriptionWireItem.Type
