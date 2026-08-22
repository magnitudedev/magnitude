import { Schema } from "effect"
import { Group, Query } from "@magnitudedev/effect-query"
import { AcnHealthResponseSchema } from "../schemas/acn-health"

/** Lifecycle-neutral health observation; executed by the transport during selection. */
const Health = Query.make("Health", {
  policy: { demand: false },
  payload: Schema.Struct({}),
  success: AcnHealthResponseSchema,
})

export const Connection = Group.make({ Health })
