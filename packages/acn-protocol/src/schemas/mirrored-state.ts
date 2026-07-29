import { Schema } from "effect"

/** Versioned authoritative state read by clients and invalidated by a watch stream. */
export const MirroredSnapshotSchema = <A, I, R>(state: Schema.Schema<A, I, R>) => Schema.Struct({
  revision: Schema.NonNegativeInt,
  state,
})

export interface MirroredSnapshot<State> {
  readonly revision: number
  readonly state: State
}

export const MirroredStateInvalidationSchema = Schema.TaggedStruct("changed", {
  id: Schema.String,
  revision: Schema.NonNegativeInt,
})
export type MirroredStateInvalidation = Schema.Schema.Type<typeof MirroredStateInvalidationSchema>
