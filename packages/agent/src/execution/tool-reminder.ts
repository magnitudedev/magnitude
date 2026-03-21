/**
 * ToolReminderTag — Effect service for tools to push contextual reminders during execution.
 *
 * Uses a Ref-backed pattern. Tools call
 * `yield* (yield* ToolReminderTag).add(text)` to enqueue reminder text that the
 * execution manager consumes after ToolExecutionEnded and surfaces in the next
 * turn's system inbox.
 */

import { Context, Effect } from 'effect'

export class ToolReminderTag extends Context.Tag('ToolReminder')<
  ToolReminderTag,
  { readonly add: (text: string) => Effect.Effect<void> }
>() {}