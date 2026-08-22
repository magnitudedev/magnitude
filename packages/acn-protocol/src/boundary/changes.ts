import { Schema } from "effect"
import { Group, Subscription } from "@magnitudedev/effect-query"
import { ProjectStoreUnavailable, SessionInspectionUnavailable } from "../errors"
import { ChangeSchema } from "../schemas/changes"

/**
 * One connection-global subscription carrying every poke the ACN publishes.
 * Clients invalidate the named queries; nothing else is derived from it.
 */
const StreamChanges = Subscription.make("StreamChanges", {
  payload: Schema.Struct({}),
  success: ChangeSchema,
  error: Schema.Union(ProjectStoreUnavailable, SessionInspectionUnavailable),
})

export const Changes = Group.make({ StreamChanges })
