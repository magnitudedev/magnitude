import { describe, test, expect } from 'bun:test'
import { buildExtractionTranscript } from '../transcript'
import type { AppEvent } from '../../events'

describe('memory transcript', () => {
  test('includes user messages verbatim and only user-targeted assistant messages', () => {
    const events = [
      {
        type: 'user_message',
        forkId: null,
        content: [{ type: 'text', text: 'Use named exports, not default exports.' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      },
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
              '<lenses><lens name="task">thinking</lens></lenses><comms><message to="user">Got it, I will update exports.</message><message to="scout-1">Please inspect files.</message></comms><actions><inspect><ref tool="fs-read"/></inspect></actions>',
          },
          {
            type: 'thinking',
            content: 'internal thoughts',
          },
        ],
        toolCalls: [],
        inspectResults: [],
        result: { success: true, turnDecision: 'finish' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
      },
    ] as unknown as AppEvent[]

    const t = buildExtractionTranscript(events)
    expect(t).toContain('Use named exports, not default exports.')
    expect(t).toContain('Got it, I will update exports.')
    expect(t).not.toContain('Please inspect files.')
    expect(t).not.toContain('<actions>')
  })

  test('includes delegation lifecycle but excludes removed proposal events', () => {
    const events = [
      {
        type: 'agent_created',
        forkId: null,
        agentId: 'a1',
        taskId: 'task1',
        agentType: 'scout',
        agentForkId: 'f1',
        prompt: 'scan',
      },
      {
        type: 'fork_started',
        forkId: 'f1',
        parentForkId: null,
        name: 'Scout',
        agentId: 'a1',
        context: 'ctx',
        mode: 'clone',
        role: 'scout',
        taskId: 'task1',
      },
      {
        type: 'fork_completed',
        forkId: null,
        childForkId: 'f1',
        result: { summary: 'done' },
      },
      {
        type: 'proposal_created',
        forkId: null,
        proposalId: 'p1',
        goal: 'Implement feature',
        criteria: {},
        attachedArtifactIds: [],
      },
    ] as unknown as AppEvent[]

    const t = buildExtractionTranscript(events)
    expect(t).toContain('agent_created')
    expect(t).toContain('fork_started')
    expect(t).toContain('fork_completed')
    expect(t).not.toContain('proposal_created')
  })

  test('truncation keeps temporal order and drops oldest lines first', () => {
    const events = [
      {
        type: 'user_message',
        forkId: null,
        content: [{ type: 'text', text: 'old-1' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      },
      {
        type: 'user_message',
        forkId: null,
        content: [{ type: 'text', text: 'mid-2' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      },
      {
        type: 'user_message',
        forkId: null,
        content: [{ type: 'text', text: 'new-3' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      },
    ] as unknown as AppEvent[]

    const t = buildExtractionTranscript(events, { maxChars: 80 })
    expect(t).not.toContain('old-1')
    expect(t).toContain('mid-2')
    expect(t).toContain('new-3')
    expect(t.indexOf('mid-2')).toBeLessThan(t.indexOf('new-3'))
  })

  test('includes multiple user messages in one assistant turn in order', () => {
    const events = [
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
              '<comms><message to="user">First message.</message><message to="user">Second message.</message></comms>',
          },
        ],
        toolCalls: [],
        inspectResults: [],
        result: { success: true, turnDecision: 'finish' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
      },
    ] as unknown as AppEvent[]

    const t = buildExtractionTranscript(events)
    expect(t).toContain('First message.')
    expect(t).toContain('Second message.')
    expect(t.indexOf('First message.')).toBeLessThan(t.indexOf('Second message.'))
  })

  test('omits assistant turn when there is no user-directed message', () => {
    const events = [
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
              '<comms><message to="scout-1">Inspect src/</message></comms><actions><inspect><ref tool="fs-tree"/></inspect></actions>',
          },
        ],
        toolCalls: [],
        inspectResults: [],
        result: { success: true, turnDecision: 'finish' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
      },
    ] as unknown as AppEvent[]

    const t = buildExtractionTranscript(events)
    expect(t).toBe('')
  })

  test('excludes subagent-directed messages from assistant turn output', () => {
    const events = [
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
              '<comms><message to="researcher-1">Gather docs.</message><message to="user">Done.</message></comms>',
          },
        ],
        toolCalls: [],
        inspectResults: [],
        result: { success: true, turnDecision: 'finish' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
      },
    ] as unknown as AppEvent[]

    const t = buildExtractionTranscript(events)
    expect(t).toContain('Done.')
    expect(t).not.toContain('Gather docs.')
  })
})