/**
 * Tool result formatter utilities for window-to-prompt conversion.
 *
 * createTruncatingFormatter: overrides large Success outputs with truncated references.
 * createAgentFormatter: wraps with <permission_rejected> for denied tool calls.
 */

import type { ToolResultPart } from '@magnitudedev/ai'
import type { ToolResultEntry, ToolResult } from '@magnitudedev/harness'
import { isImageValue, type ToolResultFormatter } from '@magnitudedev/harness'
import { describeShape, estimateText } from '../../truncation'
import { TRUNCATION_TOKEN_LIMIT } from '../../constants'

// ---------------------------------------------------------------------------
// Truncation override for large Success outputs
// ---------------------------------------------------------------------------

export function formatTruncatedSuccess(
  entry: ToolResultEntry & { result: Extract<ToolResult, { _tag: 'Success' }> },
  turnId: string,
  estimatedTokens: number,
): readonly ToolResultPart[] {
  const resultPath = `$M/results/${turnId}_${entry.toolCallId}.json`
  const shapeSummary = describeShape(entry.result.output)
  const text = [
    `<truncated path="${resultPath}" estimated_tokens="${estimatedTokens}">`,
    shapeSummary,
    `</truncated>`,
  ].join('\n')
  return [{ _tag: 'TextPart', text }]
}

/**
 * Create a truncating formatter that overrides large Success outputs.
 * Delegates everything else to the default formatter.
 */
export function createTruncatingFormatter(
  defaultFormat: ToolResultFormatter,
  turnId: string,
): ToolResultFormatter {
  return (entry: ToolResultEntry): readonly ToolResultPart[] => {
    const result = entry.result
    if (result._tag === 'Success' && result.output !== undefined && !isImageValue(result.output)) {
      try {
        const serialized = JSON.stringify(result.output, null, 2)
        const estimatedTokens = estimateText(serialized)
        if (estimatedTokens > TRUNCATION_TOKEN_LIMIT) {
          return formatTruncatedSuccess(entry as ToolResultEntry & { result: Extract<ToolResult, { _tag: 'Success' }> }, turnId, estimatedTokens)
        }
      } catch {
        // fall through to default
      }
    }
    return defaultFormat(entry)
  }
}

/**
 * Wrap the harness formatter with agent-specific overrides.
 * Adds domain-specific <permission_rejected> formatting for denied tool calls.
 */
export function createAgentFormatter(
  harnessFormat: ToolResultFormatter,
): ToolResultFormatter {
  return (entry: ToolResultEntry): readonly ToolResultPart[] => {
    if (entry.result._tag === 'Denied') {
      const message = typeof entry.result.denial === 'string'
        ? entry.result.denial
        : String(entry.result.denial)
      return [{ _tag: 'TextPart', text:
        `<permission_rejected>\n` +
        `<reason>${message}</reason>\n` +
        `This restriction exists to prevent accidental or catastrophic operations. Do not try to work around it — respect the intent of the restriction rather than finding methods that bypass the check. Provide the command to the user if you need them to run it.\n` +
        `</permission_rejected>`
      }]
    }
    return harnessFormat(entry)
  }
}
