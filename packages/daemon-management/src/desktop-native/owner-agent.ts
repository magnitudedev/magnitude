import { Effect } from "effect"
import type { OwnerReply } from "@magnitudedev/acn-protocol"

type Handled = { readonly reply: OwnerReply; readonly afterReply: Effect.Effect<void> }

export const ownerDone = (afterReply: Effect.Effect<void> = Effect.void): Handled => ({ reply: { _tag: "Done" }, afterReply })
export const ownerUnsupported: Effect.Effect<Handled> = Effect.succeed({ reply: { _tag: "Unsupported" }, afterReply: Effect.void })
/** An owner operation's outcome as a reply; its failure message is already safe to show. */
export const ownerResult = <E extends { readonly message: string }, R>(effect: Effect.Effect<unknown, E, R>, afterReply: Effect.Effect<void> = Effect.void): Effect.Effect<Handled, never, R> =>
  effect.pipe(
    Effect.as(ownerDone(afterReply)),
    Effect.catchAll(error => Effect.succeed<Handled>({ reply: { _tag: "Failed", message: error.message }, afterReply: Effect.void })),
  )
