import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import {
  MirroredSnapshotSchema,
  MirroredStateWatchEventSchema,
} from "../schemas/mirrored-state"

/** Defines the complete RPC contract and client reactivity identity for mirrored state. */
export const defineMirroredState = <
  const Id extends string,
  State,
  StateEncoded,
  StateRequirements,
  Error,
  ErrorEncoded,
  ErrorRequirements,
>(id: Id, options: {
  readonly stateSchema: Schema.Schema<State, StateEncoded, StateRequirements>
  readonly errorSchema: Schema.Schema<Error, ErrorEncoded, ErrorRequirements>
}) => {
  const snapshotSchema = MirroredSnapshotSchema(options.stateSchema)
  const emptyPayload = {}
  return {
    ...options,
    id,
    snapshotSchema,
    getPayload: emptyPayload,
    getRpc: Rpc.make(id, {
      payload: Schema.Struct({}),
      success: snapshotSchema,
      error: options.errorSchema,
    }),
  }
}

export const WatchMirroredStates = Rpc.make("WatchMirroredStates", {
  payload: Schema.Struct({}),
  success: MirroredStateWatchEventSchema,
  error: Schema.Never,
  stream: true,
})
