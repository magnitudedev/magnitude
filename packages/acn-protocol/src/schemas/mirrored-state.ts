import { Schema } from "effect"

/** Versioned authoritative state read by clients; a change poke names its query and revision. */
export const MirroredSnapshotSchema = <A, I, R>(state: Schema.Schema<A, I, R>) => Schema.Struct({
  revision: Schema.NonNegativeInt,
  state,
})

export interface MirroredSnapshot<State> {
  readonly revision: number
  readonly state: State
}
