/**
 * InterruptPubSub
 *
 * Broadcast channel for interrupt signals.
 * Workers race their handlers against this signal for automatic interruption.
 */

import { Context, Layer, PubSub, Effect } from 'effect'

/**
 * Payload is the forkId being interrupted (null = root).
 * Consumers filter by forkId to only interrupt matching handlers.
 */
export const InterruptPubSub = Context.GenericTag<PubSub.PubSub<string | null>>('InterruptPubSub')

export const InterruptPubSubLive = Layer.effect(
  InterruptPubSub,
  PubSub.unbounded<string | null>()
)
