import { Schema } from "effect"

/** Read from the running packaged application's native SMAppService, not UI text. */
export const MacLoginRegistration = Schema.Struct({ executable: Schema.NonEmptyString,
  status: Schema.Literal("enabled"), packaged: Schema.Literal(true) })
export type MacLoginRegistration = typeof MacLoginRegistration.Type
