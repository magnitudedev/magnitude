import { Context, Effect, Layer, Scope } from "effect"
import { AcnRpcClientTag } from "@magnitudedev/sdk"
import { runMirroredStateInvalidationWatch } from "./mirrored-state-invalidation"

export interface EffectQueryInvalidations {
  readonly register: (
    mirrorId: string,
    invalidate: () => Effect.Effect<void>,
  ) => Effect.Effect<void, never, Scope.Scope>
}

export const EffectQueryInvalidations = Context.GenericTag<EffectQueryInvalidations>(
  "client/EffectQueryInvalidations",
)

export const EffectQueryInvalidationsLive = Layer.scoped(
  EffectQueryInvalidations,
  Effect.gen(function* () {
    const rpc = yield* AcnRpcClientTag
    const callbacks = new Map<string, Set<() => Effect.Effect<void>>>()
    const run = (registered: Iterable<() => Effect.Effect<void>>) =>
      Effect.forEach(registered, (invalidate) => invalidate(), { discard: true })

    yield* runMirroredStateInvalidationWatch(
      rpc,
      () => run([...callbacks.values()].flatMap((entries) => [...entries])),
      (event) => run(callbacks.get(event.id) ?? []),
    ).pipe(Effect.forkScoped)

    return {
      register: (mirrorId, invalidate) => Effect.acquireRelease(
        Effect.sync(() => {
          const registered = callbacks.get(mirrorId) ?? new Set()
          registered.add(invalidate)
          callbacks.set(mirrorId, registered)
        }),
        () => Effect.sync(() => {
          const registered = callbacks.get(mirrorId)
          registered?.delete(invalidate)
          if (registered?.size === 0) callbacks.delete(mirrorId)
        }),
      ).pipe(Effect.zipRight(invalidate())),
    }
  }),
)
