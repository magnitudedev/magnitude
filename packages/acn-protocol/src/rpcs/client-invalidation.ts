import { Schema } from "effect"
import { ProjectStoreUnavailable, SessionInspectionUnavailable } from "../errors"
import { ClientInvalidationSchema } from "../schemas/client-invalidation"
import { makeAcnSubscriptionRpc } from "./subscription"

/** One transport subscription for connection-global invalidation events. */
export const StreamClientInvalidations = makeAcnSubscriptionRpc(
  "StreamClientInvalidations",
  {
    payload: Schema.Struct({}),
    success: ClientInvalidationSchema,
    error: Schema.Union(ProjectStoreUnavailable, SessionInspectionUnavailable),
  },
)
