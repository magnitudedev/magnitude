/**
 * Static prompt text fragments injected into LLM context.
 */


/** Message shown to the LLM when its turn is interrupted by the user */
export const INTERRUPT_MESSAGE = 'Your previous turn was interrupted by the user before completion. Any partial work from that turn is shown above.'


/** Wrapper tag for compacted summary in LLM context */
export function compactionSummaryTag(summary: string): string {
  return `<compaction_summary>\n${summary}\n</compaction_summary>`
}
