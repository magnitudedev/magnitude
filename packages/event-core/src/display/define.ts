/**
 * Display.define() - UI State Composition
 *
 * Takes a single projection source. For multiple sources, compose projections first.
 */

import { Effect, Stream, SubscriptionRef, Context, Duration, Layer } from 'effect'
import { HydrationContext } from '../core/hydration-context'
import type { ProjectionInstance } from '../projection/define'

export interface DisplayInstance<State> {
  state: SubscriptionRef.SubscriptionRef<State>
  stream: Stream.Stream<State>
}

export function define<S, DerivedState>(config: {
  name: string
  source: Context.Tag<ProjectionInstance<S>, ProjectionInstance<S>>
  derive: (projection: ProjectionInstance<S>) => Effect.Effect<DerivedState, never, never>
  options?: {
    debounce?: Duration.DurationInput
  }
}) {
  const serviceName = `${config.name}Display`
  const Tag = Context.GenericTag<DisplayInstance<DerivedState>>(serviceName)

  const Live = Layer.scoped(Tag, Effect.gen(function* () {
    const hydration = yield* HydrationContext
    const projection = yield* config.source

    const derivedStream = projection.state.changes.pipe(
      Stream.mapEffect(() => config.derive(projection)),
      Stream.filterEffect(() =>
        hydration.isHydrating().pipe(Effect.map(h => !h))
      ),
      config.options?.debounce ? Stream.debounce(config.options.debounce) : (s) => s
    )

    const initialDerived = yield* config.derive(projection)
    const stateRef = yield* SubscriptionRef.make(initialDerived)

    yield* Stream.runForEach(derivedStream, (val) =>
      SubscriptionRef.set(stateRef, val)
    ).pipe(Effect.forkScoped)

    return {
      state: stateRef,
      stream: stateRef.changes
    }
  }))

  return {
    Tag,
    Layer: Live
  }
}
