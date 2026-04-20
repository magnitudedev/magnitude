/**
 * Mact Parser — new architecture.
 *
 * Consumes Token events from the tokenizer and emits TurnEngineEvent directly.
 * Unlike the old parser.ts which emitted intermediate ParseEvent/StructuralEvent,
 * this parser emits consumer-facing events:
 *   - ToolInputStarted, ToolInputFieldChunk, ToolInputFieldComplete, ToolInputReady
 *   - ToolInputParseError (for unknown tools, unknown params, missing required fields)
 *   - LensStart/Chunk/End, MessageStart/Chunk/End, TurnControl
 *   - ProseChunk/ProseEnd, StructuralParseError
 *
 * The parser integrates jsonish for JSON-type fields:
 *   - One StreamingJsonParser per JSON field, created at ParameterOpen
 *   - currentPath from jsonish is used to compute ToolInputFieldChunk.path
 *   - At ParameterClose, jsonish finalizes and coerces the value
 *
 * Parse-time validation:
 *   - Unknown tool → ToolInputParseError { UnknownTool }, dead InvokeFrame
 *   - Unknown parameter → ToolInputParseError { UnknownParameter }, dead ParameterFrame
 *   - Missing required fields → ToolInputParseError { MissingRequiredField } at InvokeClose
 *   - Incomplete invoke (never closed) → ToolInputParseError { IncompleteTool } at end()
 */

import { createTokenizer } from '../tokenizer'
import { createStackMachine, type Op } from '../machine'
import { createStreamingJsonParser } from '../jsonish/parser'
import { coerceToStreamingPartial } from '../jsonish/coercer'
import { deriveParameters } from '../execution/parameter-schema'
import { generateToolInterface, printAst } from '@magnitudedev/tools'
import type {
  Token,
  TurnEngineEvent,
  RegisteredTool,
  ParseErrorDetail,
  DeepPaths,
} from '../types'
import type {
  Frame,
  ProseFrame,
  ThinkFrame,
  MessageFrame,
  InvokeFrame,
  ParameterFrame,
  FilterFrame,
  FieldState,
  FieldType,
} from './types'
import type { ParserConfig } from './types'

export type { ParserConfig } from './types'

// =============================================================================
// Character helpers
// =============================================================================

function isWhitespace(ch: string): boolean {
  return ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r'
}

function isNewline(ch: string): boolean {
  return ch === '\n'
}

function countLeadingWhitespace(text: string): number {
  for (let i = 0; i < text.length; i++) {
    if (!isWhitespace(text[i])) return i
  }
  return text.length
}

function countTrailingWhitespace(text: string): number {
  for (let i = text.length - 1; i >= 0; i--) {
    if (!isWhitespace(text[i])) return text.length - 1 - i
  }
  return text.length
}

function countNewlines(text: string): number {
  let count = 0
  for (let i = 0; i < text.length; i++) {
    if (isNewline(text[i])) count++
  }
  return count
}

function isAllWhitespace(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    if (!isWhitespace(text[i])) return false
  }
  return true
}

function stripLeadingWhitespace(text: string): string {
  const count = countLeadingWhitespace(text)
  return count === 0 ? text : text.slice(count)
}

function stripTrailingWhitespace(text: string): string {
  const count = countTrailingWhitespace(text)
  return count === 0 ? text : text.slice(0, text.length - count)
}

// =============================================================================
// Coalescing layer
// =============================================================================

type Writable<T> = { -readonly [K in keyof T]: T[K] }

interface CoalescingBuffer {
  key: string
  event: Writable<TurnEngineEvent>
}

function classifyEvent(event: TurnEngineEvent): string | null {
  switch (event._tag) {
    case 'LensChunk': return 'lens'
    case 'MessageChunk': return `message:${event.id}`
    case 'ProseChunk': return 'prose'
    case 'ToolInputFieldChunk': return `field:${event.toolCallId}:${event.field}`
    default: return null
  }
}

function mergeEvent(target: Writable<TurnEngineEvent>, source: TurnEngineEvent): void {
  if ('text' in target && 'text' in source) {
    (target as { text: string }).text += (source as { text: string }).text
  }
}

// =============================================================================
// Prose helpers
// =============================================================================

function appendProse(top: ProseFrame, text: string): Op<Frame, TurnEngineEvent>[] {
  const ops: Op<Frame, TurnEngineEvent>[] = []

  if (!top.hasContent) {
    const stripped = stripLeadingWhitespace(text)
    if (stripped.length === 0) {
      ops.push({ type: 'replace', frame: { ...top, pendingNewlines: top.pendingNewlines + countNewlines(text) } })
      return ops
    }
    ops.push({ type: 'replace', frame: { ...top, body: stripped, hasContent: true, pendingNewlines: 0 } })
    ops.push({ type: 'emit', event: { _tag: 'ProseChunk', text: stripped } })
    return ops
  }

  if (top.pendingNewlines > 0) {
    const prefix = '\n'.repeat(top.pendingNewlines)
    ops.push({ type: 'replace', frame: { ...top, body: top.body + prefix, pendingNewlines: 0 } })
    ops.push({ type: 'emit', event: { _tag: 'ProseChunk', text: prefix } })
  }

  ops.push({ type: 'replace', frame: { ...top, body: top.body + text, pendingNewlines: 0 } })
  ops.push({ type: 'emit', event: { _tag: 'ProseChunk', text } })
  return ops
}

function endTopProse(top: ProseFrame): Op<Frame, TurnEngineEvent>[] {
  const trimmed = stripTrailingWhitespace(top.body)
  if (trimmed.length === 0 && !top.hasContent) {
    return [{ type: 'replace', frame: { type: 'prose', body: '', pendingNewlines: 0, hasContent: false } }]
  }
  return [
    { type: 'emit', event: { _tag: 'ProseEnd', content: trimmed } },
    { type: 'replace', frame: { type: 'prose', body: '', pendingNewlines: 0, hasContent: false } },
  ]
}

// =============================================================================
// Message helpers
// =============================================================================

function appendMessage(top: MessageFrame, text: string): Op<Frame, TurnEngineEvent>[] {
  const ops: Op<Frame, TurnEngineEvent>[] = []

  if (isAllWhitespace(text) && countNewlines(text) === text.length) {
    ops.push({ type: 'replace', frame: { ...top, pendingNewlines: top.pendingNewlines + text.length } })
    return ops
  }

  let segment = text
  let trailingNewlines = 0

  if (top.content.length === 0) {
    segment = stripLeadingWhitespace(segment)
  }

  for (let i = segment.length - 1; i >= 0; i--) {
    if (isNewline(segment[i])) trailingNewlines++
    else break
  }
  if (trailingNewlines > 0) {
    segment = segment.slice(0, segment.length - trailingNewlines)
  }

  const prefix = top.pendingNewlines > 0 && top.content.length > 0
    ? '\n'.repeat(top.pendingNewlines)
    : ''

  const full = prefix + segment
  const nextBody = full.length > 0 ? top.content + full : top.content

  ops.push({ type: 'replace', frame: { ...top, content: nextBody, pendingNewlines: trailingNewlines } })
  if (full.length > 0) {
    ops.push({ type: 'emit', event: { _tag: 'MessageChunk', id: top.id, text: full } })
  }
  return ops
}

// =============================================================================
// Coerce scalar field value
// =============================================================================

function coerceScalarValue(rawValue: string, fieldType: FieldType): { value: unknown; ok: boolean } {
  const trimmed = rawValue.trim()
  switch (fieldType) {
    case 'string':
      return { value: trimmed, ok: true }
    case 'number': {
      const num = parseFloat(trimmed)
      if (isNaN(num)) return { value: trimmed, ok: false }
      return { value: num, ok: true }
    }
    case 'boolean': {
      const lower = trimmed.toLowerCase()
      if (lower === 'true' || lower === '1') return { value: true, ok: true }
      if (lower === 'false' || lower === '0') return { value: false, ok: true }
      return { value: trimmed, ok: false }
    }
    default:
      return { value: trimmed, ok: true }
  }
}

// =============================================================================
// Parser Implementation
// =============================================================================

export interface MactParser {
  pushToken(token: Token): void
  end(): void
  /** Consume and return all pending events, clearing the internal buffer */
  drain(): readonly TurnEngineEvent[]
}

export function createParser(config: ParserConfig): MactParser {
  let idCounter = 0
  const generateId = config.generateId ?? (() => `mact-${++idCounter}-${Date.now().toString(36)}`)

  // Derive tool schemas eagerly from the registry
  const toolSchemas = new Map<string, ReturnType<typeof deriveParameters>>()
  for (const [tagName, registeredTool] of config.tools) {
    try {
      const schema = deriveParameters(registeredTool.tool.inputSchema.ast)
      toolSchemas.set(tagName, schema)
    } catch {
      // If schema derivation fails, tool parameters won't be validated
    }
  }

  const events: TurnEngineEvent[] = []
  let coalescingBuffer: CoalescingBuffer | null = null
  let deferredYieldTarget: 'user' | 'tool' | 'worker' | 'parent' | null = null
  let postYieldHasContent = false

  function flushCoalescing(): void {
    if (coalescingBuffer === null) return
    events.push(coalescingBuffer.event as TurnEngineEvent)
    coalescingBuffer = null
  }

  function emit(event: TurnEngineEvent): void {
    const key = classifyEvent(event)

    if (key === null) {
      flushCoalescing()
      events.push(event)
      return
    }

    if (coalescingBuffer !== null && coalescingBuffer.key === key) {
      mergeEvent(coalescingBuffer.event, event)
      return
    }

    flushCoalescing()
    coalescingBuffer = { key, event: { ...event } as Writable<TurnEngineEvent> }
  }

  const machine = createStackMachine<Frame, TurnEngineEvent>(
    { type: 'prose', body: '', pendingNewlines: 0, hasContent: false },
    emit,
  )

  function getCurrentFrame<T extends Frame['type']>(type: T): Extract<Frame, { type: T }> | undefined {
    for (let i = machine.stack.length - 1; i >= 0; i--) {
      const frame = machine.stack[i]
      if (frame.type === type) return frame as Extract<Frame, { type: T }>
    }
    return undefined
  }

  function endCurrentProse(): void {
    const top = machine.peek()
    if (top?.type === 'prose') {
      machine.apply(endTopProse(top))
    }
  }

  // ---------------------------------------------------------------------------
  // Invoke helpers
  // ---------------------------------------------------------------------------

  function getRegisteredTool(toolTag: string): RegisteredTool | undefined {
    return config.tools.get(toolTag)
  }

  function getCorrectToolShape(toolTag: string): string | undefined {
    const registered = getRegisteredTool(toolTag)
    if (!registered) return undefined
    try {
      const result = generateToolInterface(registered.tool, registered.groupName ?? 'tools', undefined, { extractCommon: false, showErrors: false })
      return result.signature
    } catch {
      return undefined
    }
  }

  function getFieldType(toolTag: string, paramName: string): FieldType {
    const schema = toolSchemas.get(toolTag)
    if (!schema) return 'unknown'
    const param = schema.parameters.get(paramName)
    if (!param) return 'unknown'
    if (param.type === 'json') return 'json'
    if (typeof param.type === 'object' && param.type._tag === 'enum') return 'string'
    return param.type as FieldType
  }

  function buildInvokeContext(invokeFrame: InvokeFrame): { tagName: string; toolName: string; group: string } {
    return { tagName: invokeFrame.toolTag, toolName: invokeFrame.toolName, group: invokeFrame.group }
  }

  // ---------------------------------------------------------------------------
  // handleOpen
  // ---------------------------------------------------------------------------

  function handleOpen(name: string, variant: string | undefined): void {
    const top = machine.peek()

    switch (name) {
      case 'think': {
        const lensName = variant ?? 'analyze'
        const currentThink = getCurrentFrame('think')
        if (currentThink) {
          const raw = `<|think:${lensName}>`
          machine.apply([
            { type: 'replace', frame: { ...currentThink, content: currentThink.content + raw, hasContent: true, pendingNewlines: 0 } },
          ])
          if (currentThink.hasContent || currentThink.pendingNewlines > 0) {
            machine.apply([{ type: 'emit', event: { _tag: 'LensChunk', text: raw } }])
          }
          return
        }
        endCurrentProse()
        machine.apply([
          { type: 'push', frame: { type: 'think', name: lensName, content: '', hasContent: false, pendingNewlines: 0 } },
          { type: 'emit', event: { _tag: 'LensStart', name: lensName } },
        ])
        break
      }

      case 'message': {
        const id = generateId()
        const to = variant ?? null
        endCurrentProse()
        machine.apply([
          { type: 'push', frame: { type: 'message', id, to, content: '', pendingNewlines: 0 } },
          { type: 'emit', event: { _tag: 'MessageStart', id, to } },
        ])
        break
      }

      case 'invoke': {
        if (!variant) {
          const raw = '<|invoke>'
          if (top?.type === 'prose') machine.apply(appendProse(top, raw))
          return
        }

        const toolCallId = generateId()
        const toolTag = variant
        const parts = toolTag.split(':')
        const group = parts.length > 1 ? parts[0] : 'default'
        const toolName = parts.length > 1 ? parts.slice(1).join(':') : toolTag

        endCurrentProse()

        const registered = getRegisteredTool(toolTag)

        if (!registered) {
          // Unknown tool — emit parse error, push dead frame
          machine.apply([
            {
              type: 'push',
              frame: {
                type: 'invoke',
                toolCallId,
                toolTag,
                toolName,
                group,
                known: false,
                dead: true,
                hasFilter: false,
                fieldStates: new Map(),
                seenParams: new Set(),
              },
            },
            {
              type: 'emit',
              event: {
                _tag: 'ToolInputParseError',
                toolCallId,
                tagName: toolTag,
                toolName,
                group,
                error: {
                  _tag: 'UnknownTool',
                  tagName: toolTag,
                  detail: `Unknown tool tag: '${toolTag}'`,
                } satisfies ParseErrorDetail,
              },
            },
          ])
        } else {
          // Known tool — push live frame and emit ToolInputStarted
          machine.apply([
            {
              type: 'push',
              frame: {
                type: 'invoke',
                toolCallId,
                toolTag,
                toolName: registered.tool.name,
                group: registered.groupName,
                known: true,
                dead: false,
                hasFilter: false,
                fieldStates: new Map(),
                seenParams: new Set(),
              },
            },
            {
              type: 'emit',
              event: {
                _tag: 'ToolInputStarted',
                toolCallId,
                tagName: toolTag,
                toolName: registered.tool.name,
                group: registered.groupName,
              },
            },
          ])
        }
        break
      }

      default: {
        const variantStr = variant ? `:${variant}` : ''
        const text = `<|${name}${variantStr}>`
        appendUnknownContent(top, text)
        break
      }
    }
  }

  // ---------------------------------------------------------------------------
  // handleClose
  // ---------------------------------------------------------------------------

  function handleClose(name: string, pipe: string | undefined): void {
    const top = machine.peek()

    // Piped close = filter start
    if (pipe) {
      if (name === 'invoke' && top?.type === 'invoke') {
        const toolCallId = top.toolCallId
        machine.apply([
          { type: 'replace', frame: { ...top, hasFilter: true } },
          { type: 'push', frame: { type: 'filter', toolCallId, filterType: pipe, query: '' } },
        ])
      }
      return
    }

    switch (name) {
      case 'think': {
        const thinkFrame = getCurrentFrame('think')
        if (thinkFrame) {
          const trimmed = stripTrailingWhitespace(thinkFrame.content)
          machine.apply([
            { type: 'emit', event: { _tag: 'LensEnd', name: thinkFrame.name, content: trimmed } },
            { type: 'pop' },
          ])
        }
        break
      }

      case 'message': {
        const msgFrame = getCurrentFrame('message')
        if (msgFrame) {
          machine.apply([
            { type: 'emit', event: { _tag: 'MessageEnd', id: msgFrame.id } },
            { type: 'pop' },
          ])
        }
        break
      }

      case 'invoke': {
        const invokeFrame = getCurrentFrame('invoke')
        if (invokeFrame) {
          finalizeInvoke(invokeFrame)
        }
        break
      }

      case 'parameter': {
        const paramFrame = getCurrentFrame('parameter')
        if (paramFrame) {
          finalizeParameter(paramFrame)
        } else {
          if (top?.type === 'prose') machine.apply(appendProse(top, '<parameter|>'))
        }
        break
      }

      case 'filter': {
        const filterFrame = getCurrentFrame('filter')
        const invokeFrame = getCurrentFrame('invoke')
        if (filterFrame) {
          machine.apply([{ type: 'pop' }])
        }
        if (invokeFrame && invokeFrame.hasFilter) {
          finalizeInvoke(invokeFrame)
        }
        break
      }

      default: {
        const text = `<${name}|>`
        appendUnknownContent(top, text)
        break
      }
    }
  }

  // ---------------------------------------------------------------------------
  // handleSelfClose
  // ---------------------------------------------------------------------------

  function handleSelfClose(name: string, variant: string | undefined): void {
    const top = machine.peek()

    switch (name) {
      case 'yield': {
        const target = (variant as 'user' | 'tool' | 'worker' | 'parent') ?? 'user'
        endCurrentProse()
        deferredYieldTarget = target
        postYieldHasContent = false
        machine.apply([{ type: 'observe' }])
        break
      }

      default: {
        const variantStr = variant ? `:${variant}` : ''
        const text = `<|${name}${variantStr}|>`
        appendUnknownContent(top, text)
        break
      }
    }
  }

  // ---------------------------------------------------------------------------
  // handleParameterOpen
  // ---------------------------------------------------------------------------

  function handleParameterOpen(paramName: string): void {
    const invokeFrame = getCurrentFrame('invoke')
    if (!invokeFrame) {
      // Parameter outside invoke — treat as content
      const top = machine.peek()
      const text = `<|parameter:${paramName}>`
      appendUnknownContent(top, text)
      return
    }

    // Dead invoke — push dead parameter frame to absorb content
    if (invokeFrame.dead) {
      machine.apply([
        {
          type: 'push',
          frame: {
            type: 'parameter',
            toolCallId: invokeFrame.toolCallId,
            paramName,
            dead: true,
            rawValue: '',
            jsonishParser: null,
            fieldType: 'unknown',
          },
        },
      ])
      return
    }

    const { tagName, toolName, group } = buildInvokeContext(invokeFrame)

    // Check for unknown parameter
    const schema = toolSchemas.get(invokeFrame.toolTag)
    const knownParams = schema ? schema.parameters : null

    if (knownParams !== null && !knownParams.has(paramName)) {
      // Unknown parameter
      machine.apply([
        {
          type: 'push',
          frame: {
            type: 'parameter',
            toolCallId: invokeFrame.toolCallId,
            paramName,
            dead: true,
            rawValue: '',
            jsonishParser: null,
            fieldType: 'unknown',
          },
        },
        {
          type: 'emit',
          event: {
            _tag: 'ToolInputParseError',
            toolCallId: invokeFrame.toolCallId,
            tagName,
            toolName,
            group,
            error: {
              _tag: 'UnknownParameter',
              toolCallId: invokeFrame.toolCallId,
              tagName,
              parameterName: paramName,
              detail: `Unknown parameter '${paramName}' for tool '${tagName}'`,
            } satisfies ParseErrorDetail,
            correctToolShape: getCorrectToolShape(tagName),
          },
        },
      ])
      return
    }

    // Check for duplicate parameter
    if (invokeFrame.seenParams.has(paramName)) {
      machine.apply([
        {
          type: 'push',
          frame: {
            type: 'parameter',
            toolCallId: invokeFrame.toolCallId,
            paramName,
            dead: true,
            rawValue: '',
            jsonishParser: null,
            fieldType: 'unknown',
          },
        },
      ])
      return
    }

    // Mark as seen
    invokeFrame.seenParams.add(paramName)

    // Determine field type and create jsonish parser if needed
    const fieldType = getFieldType(invokeFrame.toolTag, paramName)
    const jsonishParser = (fieldType === 'json') ? createStreamingJsonParser() : null

    // Initialize field state in invoke frame
    invokeFrame.fieldStates.set(paramName, {
      paramName,
      rawValue: '',
      coercedValue: undefined,
      errored: false,
      errorDetail: undefined,
      complete: false,
    })

    machine.apply([
      {
        type: 'push',
        frame: {
          type: 'parameter',
          toolCallId: invokeFrame.toolCallId,
          paramName,
          dead: false,
          rawValue: '',
          jsonishParser,
          fieldType,
        },
      },
    ])
  }

  // ---------------------------------------------------------------------------
  // handleParameterClose
  // ---------------------------------------------------------------------------

  function handleParameterClose(): void {
    const paramFrame = getCurrentFrame('parameter')
    if (!paramFrame) {
      const top = machine.peek()
      if (top?.type === 'prose') machine.apply(appendProse(top, '<parameter|>'))
      return
    }
    finalizeParameter(paramFrame)
  }

  // ---------------------------------------------------------------------------
  // finalizeParameter — emit ToolInputFieldComplete, update invoke field state
  // ---------------------------------------------------------------------------

  function finalizeParameter(paramFrame: ParameterFrame): void {
    if (paramFrame.dead) {
      machine.apply([{ type: 'pop' }])
      return
    }

    const invokeFrame = getCurrentFrame('invoke')

    // Finalize jsonish parser if present
    if (paramFrame.jsonishParser !== null) {
      paramFrame.jsonishParser.end()
    }

    // Coerce value
    let coercedValue: unknown
    let errored = false
    let errorDetail: string | undefined

    if (paramFrame.jsonishParser !== null) {
      // JSON field — coerce via jsonish coercer
      const partial = paramFrame.jsonishParser.partial
      if (partial === undefined) {
        // Empty JSON field — use raw value
        coercedValue = paramFrame.rawValue.trim() || undefined
      } else {
        const invokeTag = invokeFrame?.toolTag ?? ''
        const schema = toolSchemas.get(invokeTag)
        const paramSchema = schema?.parameters.get(paramFrame.paramName)
        if (paramSchema) {
          const registered = getRegisteredTool(invokeTag)
          if (registered) {
            try {
              const result = coerceToStreamingPartial(partial, registered.tool.inputSchema.ast)
              coercedValue = result?.value
            } catch {
              // Coercion failed — use raw
              coercedValue = paramFrame.rawValue
              errored = true
              errorDetail = 'JSON coercion failed'
            }
          } else {
            coercedValue = paramFrame.rawValue
          }
        } else {
          coercedValue = paramFrame.rawValue
        }
      }
    } else {
      // Scalar field
      const { value, ok } = coerceScalarValue(paramFrame.rawValue, paramFrame.fieldType)
      coercedValue = value
      if (!ok) {
        errored = true
        errorDetail = `Cannot coerce '${paramFrame.rawValue}' to ${paramFrame.fieldType}`
      }
    }

    // Update invoke field state
    if (invokeFrame) {
      const fieldState = invokeFrame.fieldStates.get(paramFrame.paramName)
      if (fieldState) {
        fieldState.rawValue = paramFrame.rawValue
        fieldState.coercedValue = coercedValue
        fieldState.errored = errored
        fieldState.errorDetail = errorDetail
        fieldState.complete = true
      }
    }

    // Emit ToolInputFieldComplete
    const path = [paramFrame.paramName] as unknown as DeepPaths<unknown>
    machine.apply([
      {
        type: 'emit',
        event: {
          _tag: 'ToolInputFieldComplete',
          toolCallId: paramFrame.toolCallId,
          field: paramFrame.paramName as string & keyof unknown,
          path,
          value: coercedValue,
        },
      },
      { type: 'pop' },
    ])
  }

  // ---------------------------------------------------------------------------
  // finalizeInvoke — check required fields, emit ToolInputReady or error
  // ---------------------------------------------------------------------------

  function finalizeInvoke(invokeFrame: InvokeFrame): void {
    if (invokeFrame.dead) {
      machine.apply([{ type: 'pop' }])
      return
    }

    const { tagName, toolName, group } = buildInvokeContext(invokeFrame)
    const schema = toolSchemas.get(invokeFrame.toolTag)

    // Check for errored fields
    for (const [, fieldState] of invokeFrame.fieldStates) {
      if (fieldState.errored) {
        machine.apply([
          {
            type: 'emit',
            event: {
              _tag: 'ToolInputParseError',
              toolCallId: invokeFrame.toolCallId,
              tagName,
              toolName,
              group,
              error: {
                _tag: 'SchemaCoercionError',
                toolCallId: invokeFrame.toolCallId,
                tagName,
                parameterName: fieldState.paramName,
                detail: fieldState.errorDetail ?? 'Coercion failed',
              } satisfies ParseErrorDetail,
              correctToolShape: getCorrectToolShape(tagName),
            },
          },
          { type: 'pop' },
        ])
        return
      }
    }

    // Check for missing required fields
    if (schema) {
      for (const [paramName, paramSchema] of schema.parameters) {
        if (paramSchema.required && !invokeFrame.fieldStates.has(paramName)) {
          machine.apply([
            {
              type: 'emit',
              event: {
                _tag: 'ToolInputParseError',
                toolCallId: invokeFrame.toolCallId,
                tagName,
                toolName,
                group,
                error: {
                  _tag: 'MissingRequiredField',
                  toolCallId: invokeFrame.toolCallId,
                  tagName,
                  parameterName: paramName,
                  detail: `Missing required field '${paramName}' for tool '${tagName}'`,
                } satisfies ParseErrorDetail,
                correctToolShape: getCorrectToolShape(tagName),
              },
            },
            { type: 'pop' },
          ])
          return
        }
      }
    }

    // Assemble input
    const input: Record<string, unknown> = {}
    for (const [paramName, fieldState] of invokeFrame.fieldStates) {
      input[paramName] = fieldState.coercedValue
    }

    machine.apply([
      {
        type: 'emit',
        event: {
          _tag: 'ToolInputReady',
          toolCallId: invokeFrame.toolCallId,
          input,
        },
      },
      { type: 'pop' },
    ])
  }

  // ---------------------------------------------------------------------------
  // handleContent
  // ---------------------------------------------------------------------------

  function handleContent(text: string): void {
    const top = machine.peek()
    if (!top) return

    switch (top.type) {
      case 'prose': {
        machine.apply(appendProse(top, text))
        break
      }

      case 'think': {
        if (!top.hasContent) {
          const stripped = stripLeadingWhitespace(text)
          if (stripped.length === 0) {
            machine.apply([
              { type: 'replace', frame: { ...top, pendingNewlines: top.pendingNewlines + countNewlines(text) } },
            ])
          } else {
            machine.apply([
              { type: 'replace', frame: { ...top, content: stripped, hasContent: true, pendingNewlines: 0 } },
              { type: 'emit', event: { _tag: 'LensChunk', text: stripped } },
            ])
          }
        } else {
          if (top.pendingNewlines > 0) {
            const prefix = '\n'.repeat(top.pendingNewlines)
            const full = prefix + text
            machine.apply([
              { type: 'replace', frame: { ...top, content: top.content + full, pendingNewlines: 0 } },
              { type: 'emit', event: { _tag: 'LensChunk', text: full } },
            ])
          } else {
            machine.apply([
              { type: 'replace', frame: { ...top, content: top.content + text } },
              { type: 'emit', event: { _tag: 'LensChunk', text } },
            ])
          }
        }
        break
      }

      case 'message': {
        machine.apply(appendMessage(top, text))
        break
      }

      case 'parameter': {
        if (top.dead) break

        // Accumulate raw value
        top.rawValue += text

        // Feed to jsonish parser if present
        if (top.jsonishParser !== null) {
          top.jsonishParser.push(text)
        }

        // Compute path: for JSON fields, use jsonish currentPath; for scalars, just [paramName]
        const jsonPath = top.jsonishParser !== null ? top.jsonishParser.currentPath : []
        const path = [top.paramName, ...jsonPath] as unknown as DeepPaths<unknown>

        machine.apply([
          {
            type: 'emit',
            event: {
              _tag: 'ToolInputFieldChunk',
              toolCallId: top.toolCallId,
              field: top.paramName as string & keyof unknown,
              path,
              delta: text,
            },
          },
        ])
        break
      }

      case 'filter': {
        top.query += text
        break
      }

      case 'invoke': {
        // Content between parameters — ignore
        break
      }
    }
  }

  // ---------------------------------------------------------------------------
  // appendUnknownContent — treat unknown tags as content in current context
  // ---------------------------------------------------------------------------

  function appendUnknownContent(top: Frame | undefined, text: string): void {
    if (!top) return
    switch (top.type) {
      case 'prose':
        machine.apply(appendProse(top, text))
        break
      case 'message':
        machine.apply(appendMessage(top, text))
        break
      case 'think':
        machine.apply([
          { type: 'replace', frame: { ...top, content: top.content + text } },
          { type: 'emit', event: { _tag: 'LensChunk', text } },
        ])
        break
      case 'parameter':
        if (!top.dead) {
          top.rawValue += text
          if (top.jsonishParser !== null) top.jsonishParser.push(text)
          const jsonPath = top.jsonishParser !== null ? top.jsonishParser.currentPath : []
          const path = [top.paramName, ...jsonPath] as unknown as DeepPaths<unknown>
          machine.apply([
            {
              type: 'emit',
              event: {
                _tag: 'ToolInputFieldChunk',
                toolCallId: top.toolCallId,
                field: top.paramName as string & keyof unknown,
                path,
                delta: text,
              },
            },
          ])
        }
        break
      case 'filter':
        top.query += text
        break
    }
  }

  // ---------------------------------------------------------------------------
  // pushToken
  // ---------------------------------------------------------------------------

  function pushToken(token: Token): void {
    if (machine.mode === 'observing') {
      if (token._tag === 'Content' && !isAllWhitespace(token.text)) {
        postYieldHasContent = true
      } else if (token._tag !== 'Content') {
        postYieldHasContent = true
      }
      return
    }
    if (machine.mode !== 'active') return

    switch (token._tag) {
      case 'Open': handleOpen(token.name, token.variant); break
      case 'Close': handleClose(token.name, token.pipe); break
      case 'SelfClose': handleSelfClose(token.name, token.variant); break
      case 'Parameter': handleParameterOpen(token.name); break
      case 'ParameterClose': handleParameterClose(); break
      case 'Content': handleContent(token.text); break
    }
  }

  // ---------------------------------------------------------------------------
  // end — flush all open frames
  // ---------------------------------------------------------------------------

  function end(): void {
    if (machine.mode === 'done') return

    flushCoalescing()

    // Emit TurnEnd from yield (with runaway detection)
    if (deferredYieldTarget !== null) {
      const termination = postYieldHasContent ? 'runaway' : 'natural'
      events.push({
        _tag: 'TurnEnd',
        result: {
          _tag: 'Success',
          turnControl: { target: deferredYieldTarget },
          termination,
        },
      })
      deferredYieldTarget = null
      machine.apply([{ type: 'done' }])
      return
    }

    // Close open parameter frame
    const paramFrame = getCurrentFrame('parameter')
    if (paramFrame) {
      finalizeParameter(paramFrame)
    }

    // Close open filter frame
    const filterFrame = getCurrentFrame('filter')
    if (filterFrame) {
      machine.apply([{ type: 'pop' }])
    }

    // Close open invoke frame — emit IncompleteTool error
    const invokeFrame = getCurrentFrame('invoke')
    if (invokeFrame) {
      if (!invokeFrame.dead) {
        machine.apply([
          {
            type: 'emit',
            event: {
              _tag: 'ToolInputParseError',
              toolCallId: invokeFrame.toolCallId,
              tagName: invokeFrame.toolTag,
              toolName: invokeFrame.toolName,
              group: invokeFrame.group,
              error: {
                _tag: 'IncompleteTool',
                toolCallId: invokeFrame.toolCallId,
                tagName: invokeFrame.toolTag,
                detail: `Invoke for '${invokeFrame.toolTag}' was never closed`,
              } satisfies ParseErrorDetail,
              correctToolShape: getCorrectToolShape(invokeFrame.toolTag),
            },
          },
          { type: 'pop' },
        ])
      } else {
        machine.apply([{ type: 'pop' }])
      }
    }

    // Close open message frame
    const msgFrame = getCurrentFrame('message')
    if (msgFrame) {
      machine.apply([
        { type: 'emit', event: { _tag: 'MessageEnd', id: msgFrame.id } },
        { type: 'pop' },
      ])
    }

    // Close open think frame
    const thinkFrame = getCurrentFrame('think')
    if (thinkFrame) {
      machine.apply([
        {
          type: 'emit',
          event: {
            _tag: 'StructuralParseError',
            error: { _tag: 'UnclosedThink', message: `Unclosed think tag: ${thinkFrame.name}` },
          },
        },
        { type: 'pop' },
      ])
    }

    // End prose
    const proseFrame = machine.peek()
    if (proseFrame?.type === 'prose') {
      const trimmed = stripTrailingWhitespace(proseFrame.body)
      if (trimmed.length > 0 || proseFrame.hasContent) {
        machine.apply([
          { type: 'emit', event: { _tag: 'ProseEnd', content: trimmed } },
          { type: 'done' },
        ])
      } else {
        machine.apply([{ type: 'done' }])
      }
    } else {
      machine.apply([{ type: 'done' }])
    }
  }

  return {
    pushToken,
    end,
    drain(): readonly TurnEngineEvent[] {
      flushCoalescing()
      const pending = [...events]
      events.length = 0
      return pending
    },
  }
}

/**
 * Convenience: create a parser and wire it to a tokenizer.
 * Returns a function that accepts text chunks and an end() function.
 */
export function createParserWithTokenizer(config: ParserConfig): {
  push(chunk: string): readonly TurnEngineEvent[]
  end(): readonly TurnEngineEvent[]
} {
  const parser = createParser(config)
  const tokenizer = createTokenizer((token) => parser.pushToken(token))

  return {
    push(chunk: string): readonly TurnEngineEvent[] {
      tokenizer.push(chunk)
      return parser.drain()
    },
    end(): readonly TurnEngineEvent[] {
      tokenizer.end()
      parser.end()
      return parser.drain()
    },
  }
}
