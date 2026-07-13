import type { Effect, Layer } from 'effect'
import type { TextPart, ImagePart } from '@magnitudedev/ai'
import type { RoleId } from '../agents/role-validation'

// ---------------------------------------------------------------------------
// Observables
// ---------------------------------------------------------------------------

export type ObservablePart = TextPart | ImagePart

export interface ObservableConfig<R = never> {
  readonly name: string
  readonly observe: () => Effect.Effect<readonly ObservablePart[], never, R>
}

export interface BoundObservable {
  readonly name: string
  readonly observe: () => Effect.Effect<readonly ObservablePart[]>
}

/**
 * Bind an ObservableConfig to a specific Effect context, producing a
 * BoundObservable whose `observe` requires no additional services.
 */
export function bindObservable<R>(
  config: ObservableConfig<R>,
  provide: <A, E>(effect: Effect.Effect<A, E, R>) => Effect.Effect<A, E>,
): BoundObservable {
  return {
    name: config.name,
    observe: () => provide(config.observe()),
  }
}

// ---------------------------------------------------------------------------
// Fork Setup Context
// ---------------------------------------------------------------------------

export interface ForkSetupContext {
  readonly forkId: string
  readonly roleId: RoleId
  readonly cwd: string
  readonly scratchpadPath: string
}
