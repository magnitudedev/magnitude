import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { ChangeSchema } from "../schemas/changes"

const StreamChanges = Rpc.make("StreamChanges", {
  payload: Schema.Struct({}),
  success: ChangeSchema,
  stream: true,
})

export const Changes = {
  streamChanges: StreamChanges,
}
