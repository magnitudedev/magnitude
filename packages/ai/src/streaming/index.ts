/**
 * Streaming field parser — public API.
 *
 * Provides incremental JSON parsing with field-level events,
 * schema validation, and typed partial access.
 */

export type { StreamingFieldParser } from "./field-parser"
export { createStreamingFieldParser } from "./field-parser"
export type { FieldEvent, StreamingPartial, StreamingLeaf } from "./types"
