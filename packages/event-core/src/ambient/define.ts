import { Effect } from 'effect'

export interface AmbientSnapshot<T> {
  readonly value: T
  readonly version: number
}

export interface AmbientDef<T, R = never> {
  readonly name: string
  readonly initial: T | Effect.Effect<T, never, R>
  readonly _type: T
}

export type AmbientValueOf<C> = C extends AmbientDef<infer T, infer _R> ? T : never
export type AmbientRequirementsOf<C> = C extends AmbientDef<infer _T, infer R> ? R : never

export function define<T, R = never>(options: {
  name: string
  initial: T | Effect.Effect<T, never, R>
}): AmbientDef<T, R> {
  return {
    name: options.name,
    initial: options.initial,
    _type: undefined as T
  }
}
