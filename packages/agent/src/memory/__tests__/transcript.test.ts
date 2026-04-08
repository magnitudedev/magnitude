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

const turnCompleted = (xml: string): AppEvent => ({
  type: 'turn_completed',
  forkId: null,
  turnId: 't1',
  chainId: 'c1',
  strategyId: 'xml-act',
  responseParts: [{ type: 'text', content: xml }],
  toolCalls: [],
  observedResults: [],
  result: { success: true, turnDecision: 'finish', evidence: '' },
  inputTokens: null,
  outputTokens: null,
  cacheReadTokens: null,
  cacheWriteTokens: null,
  providerId: null,
  modelId: null,
})

describe('memory transcript', () => {
  test('includes user messages verbatim and only user-targeted assistant messages', () => {
    const events: AppEvent[] = [
      userMessage('Use named exports, not default exports.', 1),
      {
        type: 'turn_completed',
        forkId: null,
        turnId: 't1',
        chainId: 'c1',
        strategyId: 'xml-act',
        responseParts: [
          {
            type: 'text',
            content:
              '<lenses><lens name="task">thinking</lens></lenses><message to="user">Got it, I will update exports.</message><message to="explorer-1">Please review the files.</message><read path="src/index.ts" observe="." />',
          },
          { type: 'thinking', content: 'internal thoughts' },
        ],
        toolCalls: [],
        observedResults: [],
        result: { success: true, turnDecision: 'finish', evidence: '' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      },
    ]

    const t = buildExtractionTranscript(events)
    expect(t).toContain('Use named exports, not default exports.')
    expect(t).toContain('Got it, I will update exports.')
    expect(t).not.toContain('Please review the files.')
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

  test('includes multiple user messages in one assistant turn in order', () => {
    const events: AppEvent[] = [
      turnCompleted('<message to="user">First message.</message><message to="user">Second message.</message>'),
    ]

    const t = buildExtractionTranscript(events)
    expect(t).toContain('First message.')
    expect(t).toContain('Second message.')
    expect(t.indexOf('First message.')).toBeLessThan(t.indexOf('Second message.'))
  })

  test('omits assistant turn when there is no user-directed message', () => {
    const events: AppEvent[] = [
      turnCompleted('<message to="explorer-1">Review src/</message><tree path="src" observe="." />'),
    ]

    const t = buildExtractionTranscript(events)
    expect(t).toBe('')
  })

  test('excludes subagent-directed messages from assistant turn output', () => {
    const events: AppEvent[] = [
      turnCompleted('<message to="explorer-1">Gather docs.</message><message to="user">Done.</message>'),
    ]

    const t = buildExtractionTranscript(events)
    expect(t).toContain('Done.')
    expect(t).not.toContain('Gather docs.')
  })
})