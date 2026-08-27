import { Group } from "@magnitudedev/effect-query"
import { MagnitudeBoundary } from "@magnitudedev/sdk"
import { LocalModelOperations } from "../local-models/operations"

/** The complete client operation graph: ACN capabilities plus client-common orchestration. */
export const MagnitudeOperations = Group.extend(MagnitudeBoundary, {
  Models: LocalModelOperations,
})
