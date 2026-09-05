import { Schema } from "effect"
import { JsonValueSchema } from "@magnitudedev/utils/schema"

/**
 * A change notification ("poke") in the client cache's identity space: it
 * names the operation whose authoritative data may have changed. It carries no
 * domain data; the receiving client rereads the named operation.
 *
 * - `operation`: the operation definition name (its Rpc tag).
 * - `key`: canonical structural key of the affected payload when the source
 *   can narrow to one entry; absent means every entry of that operation.
 */
export const ChangeSchema = Schema.Struct({
  operation: Schema.String,
  key: Schema.optional(JsonValueSchema),
})
export type Change = typeof ChangeSchema.Type
