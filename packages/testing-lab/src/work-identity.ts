import { Schema } from "effect"

/** Build and test work may use the same target but never share attempt authority. */
export const WorkId = Schema.String.pipe(Schema.pattern(/^(?:build|test):[a-z0-9:.-]+$/), Schema.brand("LabWorkId"))
export type WorkId = typeof WorkId.Type
