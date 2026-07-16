import { Schema } from "effect"

/** Versioned authoritative value read by clients and invalidated by a watch stream. */
export const MirroredSnapshotSchema = <A, I, R>(state: Schema.Schema<A, I, R>) => Schema.Struct({
  revision: Schema.NonNegativeInt,
  state,
})

export interface MirroredSnapshot<State> {
  readonly revision: number
  readonly state: State
}

export const MirroredResourceInvalidationSchema = Schema.TaggedStruct("changed", {
  revision: Schema.NonNegativeInt,
})
export type MirroredResourceInvalidation = Schema.Schema.Type<typeof MirroredResourceInvalidationSchema>
