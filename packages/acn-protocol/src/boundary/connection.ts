import { Schema } from "effect"
import { Group, Query } from "@magnitudedev/effect-query"
import { AcnHealthResponseSchema } from "../schemas/acn-health"

/** Health observation executed by the transport during selection. */
const Health = Query.make("Health", {
  payload: Schema.Struct({}),
  success: AcnHealthResponseSchema,
})

export const Connection = Group.make({ Health })
