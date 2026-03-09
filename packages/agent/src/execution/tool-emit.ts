/**
 * ToolEmitTag — Effect service for tools to push display data during execution.
 *
 * Replaces js-act's Emit service. Tools call `yield* (yield* ToolEmitTag).emit(data)`
 * to send display data (diffs, search results, etc.) to the CLI UI.
 *
 * The execution manager provides this service via fork layers, backed by a Ref
 * that is reset before each tool execution and consumed on ToolExecutionEnded.
 */

import { Context, Effect } from 'effect'
import type { ToolDisplay } from '../events'

export class ToolEmitTag extends Context.Tag('ToolEmit')<
  ToolEmitTag,
  { readonly emit: (value: ToolDisplay) => Effect.Effect<void> }
>() {}
