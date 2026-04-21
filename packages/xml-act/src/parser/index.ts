/**
 * Mact Parser — context-aware architecture.
 *
 * Each frame carries a `validTags` set. Only tokens whose name is in the current
 * frame's validTags are treated as structural; everything else becomes literal content.
 * This prevents `<div>`, `<strong>`, etc. inside a message from being parsed as tags.
 *
 * Emits ToolParseError and StructuralParseError events.
 * FilterReady is surfaced via an onFilterReady callback (not in TurnEngineEvent).
 */

import { createStackMachine } from '../machine'
import { deriveParameters } from '../engine/parameter-schema'
import type {
  TurnEngineEvent,
  FilterReady,
  StructuralParseError,
  ToolParseError,
} from '../types'
import type {
  Frame,
  InvokeFrame,
} from './types'
import type { ParserConfig } from './types'
import { PROSE_VALID_TAGS } from './types'
import { classifyEvent, mergeEvent, type CoalescingBuffer } from './coalesce'
import { endTopProse } from './content'
import { flushAllFrames } from './flush'
import { finalizeInvoke, type InvokeContext } from './handlers/invoke'
import { pushToken, type DispatchContext } from './dispatch'

export type { ParserConfig } from './types'

// =============================================================================
// Parser
// =============================================================================

export interface MactParser {
  pushToken(token: import('../types').Token): void
  end(): void
  drain(): readonly TurnEngineEvent[]
}

export function createParser(
  config: ParserConfig,
  onFilterReady?: (event: FilterReady) => void,
): MactParser {
  let idCounter = 0
  const generateId = config.generateId ?? (() => `mact-${++idCounter}-${Date.now().toString(36)}`)

  // Derive tool schemas eagerly
  const toolSchemas = new Map<string, ReturnType<typeof deriveParameters>>()
  for (const [tagName, registeredTool] of config.tools) {
    try {
      const schema = deriveParameters(registeredTool.tool.inputSchema.ast)
      toolSchemas.set(tagName, schema)
    } catch {
      // schema derivation failed — tool parameters won't be validated
    }
  }

  const events: TurnEngineEvent[] = []
  let coalescingBuffer: CoalescingBuffer | null = null

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
    coalescingBuffer = { key, event: { ...event } as CoalescingBuffer['event'] }
  }

  function emitStructuralError(error: StructuralParseError): void {
    emit({ _tag: 'StructuralParseError', error })
  }

  function emitToolError(
    error: ToolParseError,
    context: { toolCallId: string; tagName: string; toolName: string; group: string; correctToolShape?: string },
  ): void {
    emit({ _tag: 'ToolParseError', error, ...context })
  }

  const machine = createStackMachine<Frame, TurnEngineEvent>(
    { type: 'prose', body: '', pendingNewlines: 0, hasContent: false, validTags: PROSE_VALID_TAGS },
    emit,
  )

  function findFrame<T extends Frame['type']>(type: T): Extract<Frame, { type: T }> | undefined {
    for (let i = machine.stack.length - 1; i >= 0; i--) {
      if (machine.stack[i].type === type) return machine.stack[i] as Extract<Frame, { type: T }>
    }
    return undefined
  }

  function endCurrentProse(): void {
    const top = machine.peek()
    if (top?.type === 'prose') {
      machine.apply(endTopProse(top) as import('../machine').Op<Frame, TurnEngineEvent>[])
    }
  }

  // InvokeContext — shared by all invoke handlers
  const invokeCtx: InvokeContext = {
    tools: config.tools,
    toolSchemas,
    endCurrentProse,
    apply: (ops) => machine.apply(ops),
    emit,
    emitStructuralError,
    emitToolError,
    findFrame,
    finalizeInvoke: null as unknown as InvokeContext['finalizeInvoke'],
    onFilterReady,
    generateId,
  }
  invokeCtx.finalizeInvoke = (frame: InvokeFrame) => finalizeInvoke(frame, invokeCtx)

  // DispatchContext — shared state for token dispatch
  const deferredYield = { target: null as 'user' | 'invoke' | 'worker' | 'parent' | null, postYieldHasContent: false }
  const dispatchCtx: DispatchContext = {
    machine,
    emit,
    emitStructuralError,
    emitToolError,
    invokeCtx,
    endCurrentProse,
    generateId,
    deferredYield,
  }

  // end — finalize all open frames top-to-bottom
  function end(): void {
    if (machine.mode === 'done') return

    flushCoalescing()

    if (deferredYield.target !== null) {
      const termination = deferredYield.postYieldHasContent ? 'runaway' : 'natural'
      events.push({
        _tag: 'TurnEnd',
        result: { _tag: 'Success', turnControl: { target: deferredYield.target }, termination },
      })
      deferredYield.target = null
      machine.apply([{ type: 'done' }])
      return
    }

    flushAllFrames({
      peek: () => machine.peek(),
      apply: (ops) => machine.apply(ops),
      emit,
      emitStructuralError,
    emitToolError,
      invokeCtx,
      getCorrectToolShape: (toolTag) => {
        const registered = config.tools.get(toolTag)
        if (!registered) return undefined
        try {
          const { generateToolInterface } = require('@magnitudedev/tools')
          const result = generateToolInterface(registered.tool, registered.groupName ?? 'tools', undefined, { extractCommon: false, showErrors: false })
          return result.signature
        } catch {
          return undefined
        }
      },
    })
  }

  return {
    pushToken: (token) => pushToken(token, dispatchCtx),
    end,
    drain(): readonly TurnEngineEvent[] {
      flushCoalescing()
      const pending = [...events]
      events.length = 0
      return pending
    },
  }
}
