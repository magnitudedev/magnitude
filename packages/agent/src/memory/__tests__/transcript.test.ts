import { describe, test, expect } from 'bun:test'
import { buildExtractionTranscript } from '../transcript'
import type { AppEvent } from '../../events'

const userMessage = (text: string, index: number): AppEvent => ({
  type: 'user_message',
  messageId: `m${index}`,
  forkId: null,
  timestamp: index,
  content: [{ type: 'text', text }],
  attachments: [],
  mode: 'text',
  synthetic: false,
  taskMode: false,
})

const assistantUserMessageEvents = (text: string, turnId = 't1'): AppEvent[] => [
  {
    type: 'message_start',
    forkId: null,
    turnId,
    id: `${turnId}-msg`,
    destination: { kind: 'user' },
  },
  {
    type: 'message_chunk',
    forkId: null,
    turnId,
    id: `${turnId}-msg`,
    text,
  },
  {
    type: 'message_end',
    forkId: null,
    turnId,
    id: `${turnId}-msg`,
  },
]

describe('memory transcript', () => {
  test('includes user messages verbatim and only user-targeted assistant messages', () => {
    const events: AppEvent[] = [
      userMessage('Use named exports, not default exports.', 1),
      ...assistantUserMessageEvents('Got it, I will update exports.'),
    ]

    const t = buildExtractionTranscript(events)
    expect(t).toContain('Use named exports, not default exports.')
    expect(t).toContain('Got it, I will update exports.')
  })

  test('includes agent creation lifecycle but excludes unrelated events', () => {
    const events: AppEvent[] = [
      {
        type: 'agent_created',
        forkId: 'f1',
        parentForkId: null,
        agentId: 'a1',
        name: 'Explorer',
        role: 'explorer',
        context: 'ctx',
        mode: 'clone',
        taskId: 'task1',
        message: 'scan',
      },
      {
        type: 'wake',
        forkId: null,
      },
    ]

    const t = buildExtractionTranscript(events)
    expect(t).toContain('agent_created')
    expect(t).not.toContain('wake')
  })

  test('truncation keeps temporal order and drops oldest lines first', () => {
    const events: AppEvent[] = [userMessage('old-1', 1), userMessage('mid-2', 2), userMessage('new-3', 3)]

    const t = buildExtractionTranscript(events, { maxChars: 80 })
    expect(t).not.toContain('old-1')
    expect(t).toContain('mid-2')
    expect(t).toContain('new-3')
    expect(t.indexOf('mid-2')).toBeLessThan(t.indexOf('new-3'))
  })

  test('includes realistic streamed plain user message chunks', () => {
    const events: AppEvent[] = [
      ...assistantUserMessageEvents('Done.'),
    ]

    const t = buildExtractionTranscript(events)
    expect(t).toContain('Done.')
  })
})
