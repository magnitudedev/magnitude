import { Schema } from 'effect'

/**
 * Erased branded type for tool keys used in event definitions.
 *
 * The precise `ToolKey` union (derived from `ToolkitKeys<typeof leaderToolkit>`
 * in toolkits.ts) cannot be imported in events.ts because it creates a
 * circular type dependency: events → toolkits → task-tools → events.
 * Type aliases in that chain (e.g. EnforceJsonSafe) cannot be deferred
 * through the cycle, causing TS2502.
 *
 * `ToolKeyErased` is a zero-dependency branded string that breaks the cycle
 * while preventing arbitrary strings from being used where tool keys are
 * expected. Convert with `toToolKeyErased()` from toolkits.ts; narrow back
 * with `isToolKey()`.
 */
export const ToolKeyErasedSchema = Schema.String.pipe(Schema.brand('ToolKeyErased'))
export type ToolKeyErased = Schema.Schema.Type<typeof ToolKeyErasedSchema>
