import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { ProjectStoreUnavailable, SessionInspectionUnavailable } from "../errors"
import { ChangeSchema } from "../schemas/changes"

const StreamChanges = Rpc.make("StreamChanges", {
  payload: Schema.Struct({}),
  success: ChangeSchema,
  error: Schema.Union(ProjectStoreUnavailable, SessionInspectionUnavailable),
  stream: true,
})

export const Changes = {
  streamChanges: StreamChanges,
}
