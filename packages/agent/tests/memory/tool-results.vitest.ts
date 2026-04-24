import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import type { ToolParseError, StructuralParseErrorEvent, ToolParseErrorEvent } from '@magnitudedev/xml-act'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getRootMemory, lastInboxMessage } from './helpers'
import { getView } from '../../src/projections/memory'
import type { AppEvent } from '../../src/events'

function renderedUserTextFromMemory(messages: Parameters<typeof getView>[0]): string {
  const rendered = getView(messages, 'UTC', 'agent')
  return rendered
    .filter(m => m.role === 'user')
    .map(m => m.content.map(p => p.type === 'text' ? p.text : '').join('\n'))
    .join('\n')
}

type Harness = Effect.Effect.Success<typeof TestHarness>
type ToolEvent = Extract<AppEvent, { type: 'tool_event' }>
type TurnCompletedEvent = Extract<AppEvent, { type: 'turn_completed' }>

function invalidToolInputEvent(args: {
  toolCallId: string
  tagName: string
  error: ToolParseError
}): ToolParseErrorEvent {
  return {
    _tag: 'ToolParseError',
    toolCallId: args.toolCallId,
    tagName: args.tagName,
    toolName: args.tagName,
    group: 'default',
    correctToolShape: '',
    error: args.error,
  }
}

function toolObservationEvent(args: {
  toolCallId: string
  tagName: string
  text: string
}): ToolEvent['event'] {
  return {
    _tag: 'ToolObservation',
    toolCallId: args.toolCallId,
    tagName: args.tagName,
    query: '.',
    content: [{ type: 'text', text: args.text }],
  }
}

function toolErrorEvent(args: {
  toolCallId: string
  tagName: string
  message: string
}): ToolEvent['event'] {
  return {
    _tag: 'ToolExecutionEnded',
    toolCallId: args.toolCallId,
    tagName: args.tagName,
    group: 'default',
    toolName: args.tagName,
    result: {
      _tag: 'Error',
      error: args.message,
    },
  }
}

type ParseErrorFixture = {
  _tag: string
  id: string
  tagName: string
  detail: string
  [key: string]: unknown
}

function exampleShapeForTag(tagName: string): string {
  switch (tagName) {
    case 'read':
      return [
        '<magnitude:invoke tool="read">',
        '<magnitude:parameter name="path">...</magnitude:parameter>',
        '<magnitude:parameter name="offset">...</magnitude:parameter> <!-- optional -->',
        '<magnitude:parameter name="limit">...</magnitude:parameter> <!-- optional -->',
        '</magnitude:invoke>',
      ].join('\n')
    case 'create-task':
      return [
        '<magnitude:invoke tool="create-task">',
        '<magnitude:parameter name="id">...</magnitude:parameter> <!-- optional -->',
        '<magnitude:parameter name="title">...</magnitude:parameter>',
        '<magnitude:parameter name="parent">...</magnitude:parameter> <!-- optional -->',
        '</magnitude:invoke>',
      ].join('\n')
    case 'agent-create':
      return [
        '<magnitude:invoke tool="agent-create">',
        '<magnitude:parameter name="agentId">...</magnitude:parameter>',
        '<magnitude:parameter name="message">...</magnitude:parameter>',
        '</magnitude:invoke>',
      ].join('\n')
    default:
      return ''
  }
}

function parseErrorFixture(error: ParseErrorFixture): ToolEvent['event'] {
  const detail: ToolParseError = {
    _tag: 'MissingRequiredField',
    toolCallId: error.id,
    tagName: error.tagName,
    parameterName: 'unknown',
    detail: error.detail,
  }

  return {
    ...invalidToolInputEvent({
      toolCallId: error.id,
      tagName: error.tagName,
      error: detail,
    }),
    correctToolShape: exampleShapeForTag(error.tagName),
  }
}

function structuralParseErrorFixture(_raw: string): StructuralParseErrorEvent {
  return {
    _tag: 'StructuralParseError',
    error: {
      _tag: 'StrayCloseTag',
      tagName: 'magnitude:message',
      detail: '',
    },
  }
}

function turnCompletedEvent(turnId: string, chainId = 'c-1'): TurnCompletedEvent {
  return {
    type: 'turn_completed',
    forkId: null,
    turnId,
    chainId,
    strategyId: 'xml-act',
    result: { success: true, turnDecision: 'idle' },
    inputTokens: null,
    outputTokens: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    providerId: null,
    modelId: null,
  }
}

function* runTurnWithToolEvents(
  h: Harness,
  args: {
    turnId: string
    nextTurnId?: string
    forkId?: string | null
    chainId?: string
    toolEvents: ReadonlyArray<{
      toolCallId: string
      toolKey: ToolEvent['toolKey']
      event: ToolEvent['event']
    }>
  },
) {
  const forkId = args.forkId ?? null
  const chainId = args.chainId ?? 'c-1'
  const nextTurnId = args.nextTurnId ?? `${args.turnId}-next`

  yield* h.send({ type: 'turn_started', forkId, turnId: args.turnId, chainId })
  for (const toolEvent of args.toolEvents) {
    yield* h.send({
      type: 'tool_event',
      forkId,
      turnId: args.turnId,
      toolCallId: toolEvent.toolCallId,
      toolKey: toolEvent.toolKey,
      event: toolEvent.event,
    })
  }
  yield* h.send(turnCompletedEvent(args.turnId, chainId))
  yield* h.send({ type: 'turn_started', forkId, turnId: nextTurnId, chainId })
}

describe('memory tool results', () => {
  it.live('single tool call renders result', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-1',
        nextTurnId: 't-2',
        toolEvents: [
          {
            toolCallId: 'tc-1',
            toolKey: 'shell',
            event: toolObservationEvent({
              toolCallId: 'tc-1',
              tagName: 'shell',
              text: '<stdout>hi</stdout>',
            }),
          },
        ],
      })

      const memory = yield* getRootMemory(h)
      const text = renderedUserTextFromMemory(memory.messages)
      expect(text).toContain('<shell')
      expect(text).toContain('<stdout>hi</stdout>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('multiple tool calls render all results', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-1',
        nextTurnId: 't-2',
        toolEvents: [
          {
            toolCallId: 'tc-a',
            toolKey: 'shell',
            event: toolObservationEvent({
              toolCallId: 'tc-a',
              tagName: 'shell',
              text: '<stdout>a</stdout>',
            }),
          },
          {
            toolCallId: 'tc-b',
            toolKey: 'shell',
            event: toolObservationEvent({
              toolCallId: 'tc-b',
              tagName: 'shell',
              text: '<stdout>b</stdout>',
            }),
          },
        ],
      })

      const memory = yield* getRootMemory(h)
      const text = renderedUserTextFromMemory(memory.messages)
      expect(text).toContain('<stdout>a</stdout>')
      expect(text).toContain('<stdout>b</stdout>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('tool error is rendered', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        result: { success: false, error: 'boom', cancelled: false },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        const tr = inbox.results.find(r => r.kind === 'turn_results')
        const err = inbox.results.find(r => r.kind === 'error')
        expect(tr).toBeUndefined()
        expect(err?.kind).toBe('error')
        if (err?.kind === 'error') {
          expect(err.message).toContain('boom')
        }
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('interrupted turn renders interrupted result', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        result: { success: false, error: 'cancelled', cancelled: true },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        expect(inbox.results.some(r => r.kind === 'interrupted')).toBe(true)
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('successful no-action turn emits no-tools notice instead of noop', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send(turnCompletedEvent('t-1'))
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      const text = renderedUserTextFromMemory(memory.messages)
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        expect(inbox.results.some(r => r.kind === 'noop')).toBe(false)
      }
      expect(text).toContain('(no tools or messages were used this turn)')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('large output shows truncation guidance', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const large = 'x'.repeat(30000)

      yield* runTurnWithToolEvents(h, {
        turnId: 't-1',
        nextTurnId: 't-2',
        toolEvents: [
          {
            toolCallId: 'tc-large',
            toolKey: 'shell',
            event: toolObservationEvent({
              toolCallId: 'tc-large',
              tagName: 'shell',
              text: large,
            }),
          },
        ],
      })

      const memory = yield* getRootMemory(h)
      const text = renderedUserTextFromMemory(memory.messages)
      expect(text).toContain('<shell observe=".">')
      expect(text).toContain('xxxxxxxx')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('missing required read attr renders parse_error presentation', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-invalid-read-attr',
        nextTurnId: 't-after-invalid-read-attr',
        toolEvents: [
          {
            toolCallId: 'tc-invalid-read-attr',
            toolKey: 'fileRead',
            event: parseErrorFixture({
              _tag: 'MissingRequiredFields',
              id: 'tc-invalid-read-attr',
              tagName: 'read',
              detail: 'missing required attribute "path".',
            }),
          },
        ],
      })

      const text = renderedUserTextFromMemory((yield* getRootMemory(h)).messages)
      expect(text).toContain('<parse_error>')
      expect(text).toContain("Missing required parameter 'unknown' for tool 'read'.")
      expect(text).toContain('Tool: read')
      expect(text).toContain('Expected:')
      expect(text).toContain('<magnitude:invoke tool="read">')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('unknown read attribute renders invalid tool input with correct shape', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-invalid-read-unknown-attr',
        nextTurnId: 't-after-invalid-read-unknown-attr',
        toolEvents: [
          {
            toolCallId: 'tc-invalid-read-unknown-attr',
            toolKey: 'fileRead',
            event: parseErrorFixture({
              _tag: 'UnknownAttribute',
              id: 'tc-invalid-read-unknown-attr',
              tagName: 'read',
              detail: 'unknown attribute "foo".',
            }),
          },
        ],
      })

      const text = renderedUserTextFromMemory((yield* getRootMemory(h)).messages)
      expect(text).toContain('<parse_error>')
      expect(text).toContain("Missing required parameter 'unknown' for tool 'read'.")
      expect(text).toContain('Tool: read')
      expect(text).toContain('Expected:')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('invalid create-task attribute value renders invalid tool input with correct shape', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-invalid-create-task-value',
        nextTurnId: 't-after-invalid-create-task-value',
        toolEvents: [
          {
            toolCallId: 'tc-invalid-create-task-value',
            toolKey: 'createTask',
            event: parseErrorFixture({
              _tag: 'InvalidAttributeValue',
              id: 'tc-invalid-create-task-value',
              tagName: 'create-task',
              detail: 'invalid value "banana" for attribute "type"; expected one of: todo | bug | chore.',
            }),
          },
        ],
      })

      const text = renderedUserTextFromMemory((yield* getRootMemory(h)).messages)
      expect(text).toContain('<parse_error>')
      expect(text).toContain('Tool: create-task')
      expect(text).toContain('Expected:')
      expect(text).toContain('<magnitude:invoke tool="create-task">')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('unexpected read body renders parse_error presentation', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-invalid-read-body',
        nextTurnId: 't-after-invalid-read-body',
        toolEvents: [
          {
            toolCallId: 'tc-invalid-read-body',
            toolKey: 'fileRead',
            event: parseErrorFixture({
              _tag: 'UnexpectedBody',
              id: 'tc-invalid-read-body',
              tagName: 'read',
              detail: 'unexpected body content.',
            }),
          },
        ],
      })

      const text = renderedUserTextFromMemory((yield* getRootMemory(h)).messages)
      expect(text).toContain('<parse_error>')
      expect(text).toContain('</parse_error>')
      expect(text).toContain("Missing required parameter 'unknown' for tool 'read'.")    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('agent-create validation failure renders invalid tool input with correct shape', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-invalid-agent-create',
        nextTurnId: 't-after-invalid-agent-create',
        toolEvents: [
          {
            toolCallId: 'tc-invalid-agent-create',
            toolKey: 'agentCreate',
            event: parseErrorFixture({
              _tag: 'ToolValidationFailed',
              id: 'tc-invalid-agent-create',
              tagName: 'agent-create',
              detail: 'missing required attribute "id" and required child tag "message".',
            }),
          },
        ],
      })

      const text = renderedUserTextFromMemory((yield* getRootMemory(h)).messages)
      expect(text).toContain('<parse_error>')
      expect(text).toContain('Tool: agent-create')
      expect(text).toContain('Expected:')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('incomplete agent-create tag renders parse_error presentation with expected example', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-incomplete-agent-create',
        nextTurnId: 't-after-incomplete-agent-create',
        toolEvents: [
          {
            toolCallId: 'tc-incomplete-agent-create',
            toolKey: 'agentCreate',
            event: parseErrorFixture({
              _tag: 'IncompleteTag',
              id: 'tc-incomplete-agent-create',
              tagName: 'agent-create',
              detail: 'tool tag was not completed before turn end.',
            }),
          },
        ],
      })

      const text = renderedUserTextFromMemory((yield* getRootMemory(h)).messages)
      expect(text).toContain('<parse_error>')
      expect(text).toContain('Tool: agent-create')
      expect(text).toContain('Expected:')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('multiple invalid tool calls in one turn all render separately as parse errors', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-multiple-invalid',
        nextTurnId: 't-after-multiple-invalid',
        toolEvents: [
          {
            toolCallId: 'tc-multiple-invalid-read',
            toolKey: 'fileRead',
            event: parseErrorFixture({
              _tag: 'MissingRequiredFields',
              id: 'tc-multiple-invalid-read',
              tagName: 'read',
              detail: 'missing required attribute "path".',
            }),
          },
          {
            toolCallId: 'tc-multiple-invalid-create-task',
            toolKey: 'createTask',
            event: parseErrorFixture({
              _tag: 'UnknownAttribute',
              id: 'tc-multiple-invalid-create-task',
              tagName: 'create-task',
              detail: 'unknown attribute "foo".',
            }),
          },
        ],
      })

      const text = renderedUserTextFromMemory((yield* getRootMemory(h)).messages)
      expect(text.match(/<parse_error>/g)?.length ?? 0).toBeGreaterThanOrEqual(2)
      expect(text).toContain('Tool: read')
      expect(text).toContain('Tool: create-task')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('text-only turns emit no-tools-or-messages notice', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-text-only', chainId: 'c-1' })
      yield* h.send(turnCompletedEvent('t-text-only'))
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-after-text-only', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const text = renderedUserTextFromMemory(memory.messages)

      expect(text).toContain('(no tools or messages were used this turn)')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('no-content turn result includes no-tools-or-messages notice', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-empty', chainId: 'c-1' })
      yield* h.send(turnCompletedEvent('t-empty'))
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-after-empty', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      const text = renderedUserTextFromMemory(memory.messages)

      expect(text).toContain('(no tools or messages were used this turn)')
      expect(text).not.toContain('<noop>')
      expect(inbox?.type).toBe('inbox')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('parse-error-only turns do not produce empty response noise', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-parse-only',
        nextTurnId: 't-after-parse-only',
        toolEvents: [
          {
            toolCallId: 'tc-parse-only',
            toolKey: 'fileRead',
            event: parseErrorFixture({
              _tag: 'UnexpectedBody',
              id: 'tc-parse-only',
              tagName: 'read',
              detail: 'unexpected body content.',
            }),
          },
        ],
      })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      const text = renderedUserTextFromMemory(memory.messages)

      expect(text).toContain('<parse_error>')
      expect(text).toContain("Missing required parameter 'unknown' for tool 'read'.")
      expect(text).not.toContain('empty response')
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        expect(inbox.results.some(r => r.kind === 'noop')).toBe(false)
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('structural parse errors are preserved and rendered', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-structural', chainId: 'c-1' })
      yield* h.send({
        type: 'response.output_text.delta',
        forkId: null,
        turnId: 't-structural',
        chainId: 'c-1',
        text: '<magnitude:message to="parent">Hi</magnitude:message>\n</magnitude:message>',
      })
      yield* h.send({
        type: 'tool_event',
        forkId: null,
        turnId: 't-structural',
        toolCallId: 'tc-structural',
        toolKey: 'fileRead',
        event: structuralParseErrorFixture('</magnitude:message>'),
      })
      yield* h.send(turnCompletedEvent('t-structural'))
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-after-structural', chainId: 'c-1' })

      const text = renderedUserTextFromMemory((yield* getRootMemory(h)).messages)
      expect(text).toContain('<parse_error>')
      expect(text).toContain('Unexpected close </magnitude:message>')
      expect(text).toContain('Location:')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('runtime tool error rendering uses tagName rather than toolKey', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* runTurnWithToolEvents(h, {
        turnId: 't-runtime-error',
        nextTurnId: 't-after-runtime-error',
        toolEvents: [
          {
            toolCallId: 'tc-runtime-error',
            toolKey: 'fileRead',
            event: toolErrorEvent({
              toolCallId: 'tc-runtime-error',
              tagName: 'read',
              message: 'permission denied',
            }),
          },
        ],
      })

      const text = renderedUserTextFromMemory((yield* getRootMemory(h)).messages)
      expect(text).toContain('<tool name="read"><error>')
      expect(text).not.toContain('<tool name="fileRead"><error>')
      expect(text).toContain('permission denied')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
