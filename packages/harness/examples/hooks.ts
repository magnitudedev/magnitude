/**
 * Example: Composable tool result formatting with ToolResultFormatter
 *
 * Shows how to create a custom ToolResultFormatter by composing
 * over the default — override specific cases, delegate the rest.
 */

import { createToolResultFormatter, type ToolResultFormatter } from '../src'
import { defineToolkit } from '../src'
import type { Toolkit } from '../src'

// Build the default formatter from a toolkit
const toolkit: Toolkit = defineToolkit({ /* ... */ })
const defaultFormat = createToolResultFormatter(toolkit)

// Compose: override specific result types, delegate the rest
const customFormat: ToolResultFormatter = (entry) => {
  const result = entry.result

  // Custom: wrap shell output in a tag
  if (result._tag === 'Success' && entry.toolName === 'shell') {
    return [{ _tag: 'TextPart' as const, text: `<shell_output>${JSON.stringify(result.output)}</shell_output>` }]
  }

  // Delegate everything else to the default
  return defaultFormat(entry)
}

// ── Hooks with Effect-based interceptors ─────────────────────────────

import type { HarnessHooks } from '../src'
import { Effect } from 'effect'

const hooksWithInterceptors: HarnessHooks = {
  // Run before every tool execution — can reject or modify input
  beforeExecute: (ctx) =>
    Effect.succeed(
      ctx.toolName === 'shell'
        ? { _tag: 'Proceed' as const, modifiedInput: ctx.input }
        : { _tag: 'Proceed' as const }
    ),

  // Run after every tool execution
  afterExecute: (ctx) =>
    Effect.sync(() => {
      console.log(`Tool ${ctx.toolName} completed with ${ctx.result._tag}`)
    }),

  // Observe every harness event
  onEvent: (event) =>
    Effect.sync(() => {
      if (event._tag === 'ToolExecutionStarted') {
        console.log(`Starting tool: ${event.toolName}`)
      }
    }),
}
