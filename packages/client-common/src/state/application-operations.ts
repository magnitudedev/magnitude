import { Group } from "@magnitudedev/effect-query"
import { AcnQueries } from "../operations"
import { LocalModelOperations } from "../local-models/operations"

/** The complete client operation graph: ACN capabilities plus client-common orchestration. */
export const MagnitudeOperations = Group.extend(AcnQueries, {
  Models: LocalModelOperations,
})
