import { Schema } from "effect"
import { ProjectStoreUnavailable, SessionInspectionUnavailable } from "../errors"
import { ChangeSchema } from "../schemas/changes"
import { InspectProject, ListProjects } from "./project"
import { GetSession, ListRecentSessionDirectories, ListSessions } from "./session"
import { acnSubscription } from "./subscription"

/** Queries whose authoritative data a project-store commit may change. */
export const projectChangeQueries: ReadonlyArray<string> = [ListProjects._tag, InspectProject._tag]
/** Queries whose authoritative data a session commit may change. */
export const sessionChangeQueries: ReadonlyArray<string> = [
  ListSessions._tag,
  ListRecentSessionDirectories._tag,
  GetSession._tag,
]

/**
 * One connection-global subscription carrying every poke the ACN publishes.
 * Clients invalidate the named queries; nothing else is derived from it.
 */
export const StreamChanges = acnSubscription("StreamChanges", {
  payload: Schema.Struct({}),
  success: ChangeSchema,
  error: Schema.Union(ProjectStoreUnavailable, SessionInspectionUnavailable),
})
