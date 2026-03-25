import { describe, it, expect } from 'bun:test'
import { createStreamingXmlParser } from '../parser'

describe('repro: comment with <-- does not corrupt prose', () => {
  it('passes the exact LLM response through as prose unchanged', () => {
    const input = `\`TurnToolCall\`:
\`\`\`ts
{
  toolKey: string
  group: string
  toolName: string
  result: ToolResult
  context?: ToolContext
  xmlOutput?: string   // already the serialized XML string
}
\`\`\`

\`ToolEvent\` (separate, streaming):
\`\`\`ts
{
  type: 'tool_event'
  forkId: string | null
  turnId: string
  toolCallId: string   // <-- ref ID lives here
  toolKey: string
  event: ToolCallEvent
  display?: ToolDisplay
}
\`\`\`

The ref ID (\`toolCallId\`) is on \`ToolEvent\` which is a streaming event, not stored on \`TurnToolCall\`. So to avoid duplication — what's your thinking? Should the ref ID just live on \`TurnToolCall\` since it's genuinely part of the tool call record, not a duplicate?`

    const allEvents: any[] = []
    const knownTags = new Set(['think', 'actions', 'shell'])
    const childTagMap = new Map<string, Set<string>>()
    const parser = createStreamingXmlParser(knownTags, childTagMap, undefined, undefined)

    const events1 = parser.processChunk(input)
    const events2 = parser.flush()
    allEvents.push(...events1, ...events2)

    const prose = allEvents.filter((e: any) => e._tag === 'ProseChunk').map((e: any) => e.text).join('')
    console.log('PROSE OUTPUT:', JSON.stringify(prose))
    expect(prose).toBe(input)
  })
})