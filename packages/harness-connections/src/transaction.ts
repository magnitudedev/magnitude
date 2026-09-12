import { Cause, Context, Effect, Exit, Fiber, Schema } from "effect"

export class ConnectionRecoveryFailed extends Schema.TaggedError<ConnectionRecoveryFailed>()("ConnectionRecoveryFailed", {
  failures: Schema.Array(Schema.String),
}) {
  override get message() { return `Connection restoration is incomplete: ${this.failures.join("; ")}` }
}

/** Compensation is registered before mutation; native adapters own partial-operation recovery. */
export interface ConnectionTransaction {
  readonly compensate: <A, E, R>(description: string, undo: Effect.Effect<A, E, R>) => Effect.Effect<void, never, R>
  readonly commit: <A, E, R>(write: Effect.Effect<A, E, R>) => Effect.Effect<A, E, R>
}
export const ConnectionTransaction = Context.GenericTag<ConnectionTransaction>("ConnectionTransaction")

export const connectionTransaction = <A, E, R>(effect: Effect.Effect<A, E, R>) => Effect.uninterruptibleMask((restore) =>
  Effect.gen(function* () {
    const compensations: { readonly description: string; readonly undo: Effect.Effect<unknown, unknown> }[] = []
    let committed = false
    const transaction: ConnectionTransaction = {
      compensate: (description, undo) => Effect.gen(function* () {
        const runtime = yield* Effect.runtime<Effect.Effect.Context<typeof undo>>()
        compensations.push({ description, undo: Effect.provide(undo, runtime) })
      }),
      commit: (write) => Effect.uninterruptible(write.pipe(Effect.tap(() => Effect.sync(() => { committed = true })))),
    }
    const result = yield* Effect.exit(restore(Effect.provideService(effect, ConnectionTransaction, transaction)))
    if (Exit.isSuccess(result)) return result.value
    if (committed) return yield* Effect.failCause(result.cause)
    const failures: string[] = []
    for (const entry of compensations.reverse()) {
      // The parent must finish recovery even when cancelled. A supervised child
      // keeps each undo's own timeouts effective without inheriting cancellation.
      const restored = yield* entry.undo.pipe(Effect.interruptible, Effect.fork, Effect.flatMap(Fiber.await))
      if (Exit.isFailure(restored)) failures.push(`${entry.description}: ${Cause.pretty(restored.cause)}`)
    }
    return yield* Effect.failCause(failures.length === 0 ? result.cause : Cause.parallel(
      result.cause, Cause.fail(new ConnectionRecoveryFailed({ failures })),
    ))
  }))
