import { Schema } from "effect"
import { MirroredStateInvalidationSchema } from "./mirrored-state"

/**
 * Transport-level invalidations shared by every interactive client.
 * These notifications carry no domain authority; each receiving domain
 * rereads its own canonical query.
 */
export const ClientInvalidationSchema = Schema.Union(
  Schema.TaggedStruct("MirroredState", {
    invalidation: MirroredStateInvalidationSchema,
  }),
  Schema.TaggedStruct("Projects", {}),
  Schema.TaggedStruct("Sessions", {}),
)
export type ClientInvalidation = typeof ClientInvalidationSchema.Type
